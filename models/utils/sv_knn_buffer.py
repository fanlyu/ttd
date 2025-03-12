import numpy as np
from utils.buffer import Buffer

class SVKNNBuffer(Buffer):
    def __init__(self, args, episodic_mem_size, in_dim, out_dim, eps_mem_batch):
        super(SVKNNBuffer, self).__init__(args, episodic_mem_size, in_dim, out_dim, eps_mem_batch)

        self.out_dim = out_dim
        self.class_range = np.arange(self.out_dim)
        self.episodic_labels_int = np.zeros(self.episodic_mem_size, dtype=int) - 1
        self.input_batch_size = 10
        self.max_num_samples = 150
        self.num_k = 3
        self.dist = 0
        self.is_adversarial_sv = True
        self.adversarial_type = 'mean'
        self.adv_coeff = 1.0

        self.is_penal = False
        self.is_offline = False
        self.is_accum_curr = True
        self.is_mir = False
        self.rng = np.random.RandomState(args.random_seed)
        # if, for the first few new task input batches, simply add them to the memory
        if self.is_accum_curr:
            self.count_curr = 0
            self.current_task_id = 0
            # if self.args.accum_size is None:
            self.accum_size = self.episodic_mem_size // self.out_dim * 10 # assuming split task with 2 classes in a task
            # else:
            #     self.accum_size = self.args.accum_size * (self.episodic_mem_size // self.out_dim)

        # if self.is_offline:
        #     self.in_dim = in_dim
        #     self.current_task_id = 0
        #     self.temp_buffer = TempSVKNNBuffer(self.args, self.episodic_mem_size, self.in_dim, self.out_dim,
        #                                    self.eps_mem_batch)

    def get_sv_mem(self, model, sess, input_batch_x, input_batch_y):
        mem_filled_so_far = min(self.examples_seen_so_far, self.episodic_mem_size)

        if self.eps_mem_batch >= self.examples_seen_so_far:
            er_mem_indices = np.arange(mem_filled_so_far)
            self.rng.shuffle(er_mem_indices)
            final_x, final_y = self.episodic_images[er_mem_indices], self.episodic_labels[er_mem_indices]
        else:
            input_batch_y_int = np.argmax(input_batch_y, axis=1)
            sv_array, train_sv_x, train_sv_y = self.knn_sv_for_get_sv_mem(model, sess, input_batch_x, input_batch_y, input_batch_y_int)
            sorted_indices_sv = np.argsort(sv_array*-1)
            final_x = train_sv_x[sorted_indices_sv][0:self.eps_mem_batch]
            final_y = train_sv_y[sorted_indices_sv][0:self.eps_mem_batch]

        return final_x, final_y

    def get_dist_mem(self, model, sess, current_x, current_y):
        mem_filled_so_far = min(self.examples_seen_so_far, self.episodic_mem_size)
        if self.eps_mem_batch >= self.examples_seen_so_far:
            er_mem_indices = np.arange(mem_filled_so_far)
            self.rng.shuffle(er_mem_indices)
            final_x, final_y = self.episodic_images[er_mem_indices], self.episodic_labels[er_mem_indices]
        else:
            # distance score: the smaller the better
            current_y_int = np.argmax(current_y, axis=1)
            dist_score, cand_x, cand_y = self.distance_score(model, sess, current_x, current_y,current_y_int)
            sorted_indices = np.argsort(dist_score)
            final_x = cand_x[sorted_indices][:self.eps_mem_batch]
            final_y = cand_y[sorted_indices][:self.eps_mem_batch]
        return final_x, final_y

    def distance_score(self, model, sess, current_x, current_y, current_y_int):
        # w.r.t. input batch
        curr_eval_set, cand_set = self.get_data_for_get_sv_mem(current_x, current_y, current_y_int, self.max_num_samples)
        df_curr_eval, df_cand = self.get_deep_features(model, sess, curr_eval_set, cand_set)
        dist_score_b = self.get_distance(df_curr_eval, df_cand, cand_set)

        # w.r.t. memory samples
        excl_idcs = cand_set[4]
        mem_eval_set = self.get_stratified_sampling_data_for_sv(current_x, current_y, current_y_int, self.max_num_samples, bool_adjust_balance=False, excl_indices=excl_idcs)
        df_mem_eval =  self._get_deep_feature(model, sess, mem_eval_set)

        dist_score_a = self.get_distance(df_mem_eval, df_cand, cand_set)

        # get masks for each class sample (for mem_eval)
        n_cand, n_eval = cand_set[3], mem_eval_set[3]
        eval_label = np.repeat(mem_eval_set[2][:, None], n_cand, axis=1)
        cand_label = np.repeat(cand_set[2][None, :], n_eval, axis=0)
        mask = np.array(eval_label == cand_label, dtype=int)

        # for distances, we retrieve points close to both same class cluster and the input batch
        if self.adversarial_type == 'max':
            # to get minimum distance from the same class samples
            dist_score_a = dist_score_a * mask
            zero_mask = dist_score_a == 0
            dist_score_a[zero_mask] = 9e10
            dist_score_a = np.amin(dist_score_a, axis=0)

            # combine with minimum distance from current input batch
            dist_score = dist_score_a + np.amin(dist_score_b, axis=0)

        elif self.adversarial_type == 'mean':
            cnt = np.sum(mask, axis=0)
            zero_where = np.where(cnt == 0)[0]
            sum_to_mem = np.sum(dist_score_a * mask, axis=0)
            cnt[zero_where] = 1.
            mean_to_mem = sum_to_mem / cnt
            dist_score = mean_to_mem + np.mean(dist_score_b, axis=0)

        elif self.adversarial_type == 'vanilla':
            dist_score = np.mean(dist_score_b, axis=0)

        cand_x = cand_set[0]
        cand_y = cand_set[1]
        return dist_score, cand_x, cand_y

    def get_distance(self, df_eval, df_cand, cand_set):
        n_test = df_eval.shape[0]
        n_train = cand_set[3]
        df_test_repeat = np.repeat(df_eval, n_train, axis=0)
        df_train_tile = np.tile(df_cand, (n_test, 1))
        distance_metric = self.get_distance_metric(df_test_repeat, df_train_tile)
        return np.reshape(distance_metric, (n_test, n_train))

    def _get_deep_feature(self, model, sess, eval_set):
        num = eval_set[0].shape[0]
        x = eval_set[0]
        y = eval_set[1]

        # compute deep features with mini-batches
        bs = 64
        num_itr = num // bs + int(num % bs > 0)
        sid = 0
        deep_features = []
        for i in range(num_itr):
            eid = sid + bs if i != num_itr - 1 else num
            batch_x, batch_y = x[sid: eid], y[sid: eid]
            feed_dict = {model.x: batch_x, model.y_: batch_y}
            if 'resnet' in model.arch:
                feed_dict.update({model.train_phase: False})
            batch_deep_features = sess.run(model.features, feed_dict=feed_dict)
            deep_features.append(batch_deep_features)
            sid = eid
        if num_itr == 1:
            deep_features = deep_features[0]
        else:
            deep_features = np.concatenate(deep_features, axis=0)
        return deep_features

    def update_sv_mem(self, batch_x, batch_y, task_id, sess, model):
        batch_y_int = np.argmax(batch_y, axis=1)
        if self.is_offline:
            if self.current_task_id != task_id:
                # new task
                self.knn_sv_for_update_sv_mem(model, sess, batch_x, batch_y, batch_y_int)
                temp_buffer_size = self.episodic_mem_size // np.unique(self.episodic_labels_int).shape[0] * 2
                self.temp_buffer = TempSVKNNBuffer(self.args, temp_buffer_size, self.in_dim, self.out_dim,
                                                   self.eps_mem_batch)
                self.current_task_id = task_id
            self.temp_buffer.update_mem(batch_x, batch_y, task_id)

        else:
            if self.episodic_mem_size <= self.examples_seen_so_far + self.input_batch_size:
                # reset current task counter to 0 when new task sample comes
                if self.is_accum_curr:
                    if self.current_task_id != task_id:
                        self.count_curr = 0
                        self.current_task_id = task_id
                    # update counter
                    if self.current_task_id == 0 and self.count_curr == 0:
                        self.count_curr = self.examples_seen_so_far
                    self.count_curr += self.input_batch_size

                sv_array, n_train, train_sample_indices = self.knn_sv_for_update_sv_mem(model, sess, batch_x, batch_y, batch_y_int)
                if self.is_accum_curr and self.current_task_id != 0 and self.count_curr <= self.accum_size:
                    n_samples = n_train
                else:
                    n_samples = n_train - self.input_batch_size
                sv_array_samples = sv_array[0:n_samples]
                sv_array_input = sv_array[n_samples:]

            for (i, (er_x, er_y, er_y_int)) in (enumerate(zip(batch_x, batch_y, batch_y_int))):
                if self.episodic_mem_size > self.examples_seen_so_far:
                    self.episodic_images[self.examples_seen_so_far] = er_x
                    self.episodic_labels[self.examples_seen_so_far] = er_y
                    self.episodic_labels_int[self.examples_seen_so_far] = er_y_int
                    self.episodic_task[self.examples_seen_so_far] = task_id

                elif self.is_accum_curr and self.count_curr <= self.accum_size:
                    # replace previous task memory samples with smallest SV with input batch
                    min_sample_idx = np.argmin(sv_array_samples)
                    idx = train_sample_indices[min_sample_idx]
                    sv_array_samples[min_sample_idx] = 9e10
                    self.episodic_images[idx] = er_x
                    self.episodic_labels[idx] = er_y
                    self.episodic_labels_int[idx] = er_y_int
                    self.episodic_task[idx] = task_id

                else:
                    sv_input = sv_array_input[i]
                    min_sample_idx = np.argmin(sv_array_samples)
                    if sv_input > sv_array_samples[min_sample_idx]:
                        idx = train_sample_indices[min_sample_idx]
                        sv_array_samples[min_sample_idx] = sv_input
                        self.episodic_images[idx] = er_x
                        self.episodic_labels[idx] = er_y
                        self.episodic_labels_int[idx] = er_y_int
                        self.episodic_task[idx] = task_id

                self.examples_seen_so_far += 1

        return

    def knn_sv_for_get_sv_mem(self, model, sess, batch_x, batch_y, batch_y_int):
        # Task B
        test_sv_b, train_sv = self.get_data_for_get_sv_mem(batch_x, batch_y, batch_y_int, self.max_num_samples)
        sv_matrix_b = self.compute_sv_knn(model, sess, test_sv_b, train_sv)

        if self.is_adversarial_sv:
            # Task A
            excl_indices = train_sv[4]
            test_sv_a = self.get_stratified_sampling_data_for_sv(batch_x, batch_y, batch_y_int, self.max_num_samples, bool_adjust_balance=False, excl_indices=excl_indices)
            sv_matrix_a = self.compute_sv_knn(model, sess, test_sv_a, train_sv)

            # Adversarial Shapley value type
            if self.adversarial_type == 'max':
                sv_array = np.amax(sv_matrix_a,axis=0) - np.amin(sv_matrix_b,axis=0)
            elif self.adversarial_type == 'sum':
                sv_array = np.sum(sv_matrix_a,axis=0) - np.sum(sv_matrix_b,axis=0)
            elif self.adversarial_type == 'mean':
                sv_array = self.adv_coeff * np.mean(sv_matrix_a,axis=0) - np.mean(sv_matrix_b,axis=0)
        else:
            sv_array = np.mean(sv_matrix_b, axis=0) * -1

        # training data
        train_sv_x = train_sv[0]
        train_sv_y = train_sv[1]
        return sv_array, train_sv_x, train_sv_y

    def knn_sv_for_update_sv_mem(self, model, sess, batch_x, batch_y, batch_y_int):
        if self.is_offline:
            mem_filled = min(self.examples_seen_so_far, self.episodic_mem_size)
            x = self.episodic_images[0:mem_filled]
            y = self.episodic_labels[0:mem_filled]

            temp_buffer_filled = min(self.temp_buffer.examples_seen_so_far, self.temp_buffer.episodic_mem_size)
            temp_buffer_x = self.temp_buffer.episodic_images[0:temp_buffer_filled]
            temp_buffer_y = self.temp_buffer.episodic_labels[0:temp_buffer_filled]

            x = np.concatenate((x, temp_buffer_x), axis=0)
            y = np.concatenate((y, temp_buffer_y), axis=0)
            y_int = np.argmax(y, axis=1)
            indices = np.arange(y_int.shape[0])

            self.examples_seen_so_far += self.temp_buffer.examples_seen_so_far
            self.temp_buffer = ()

            test_sv = self.get_data_in_list(x, y, y_int, indices)
            train_sv = test_sv

            sv_matrix = self.compute_sv_knn(model, sess, test_sv, train_sv)

            n_dim = sv_matrix.shape[0]
            # diag_mask = (np.eye(n_dim) - 1) * -1
            # sv_matrix_zero_diag = np.multiply(sv_matrix, diag_mask)

            sv_array_mean = np.mean(sv_matrix, axis=0)
            ind = np.argsort(sv_array_mean * -1)[0:self.episodic_mem_size]

            self.episodic_images = x[ind]
            self.episodic_labels = y[ind]
            self.episodic_labels_int = y_int[ind]

            return
        else:
            test_sv, train_sv = self.get_data_for_update_sv_mem(batch_x, batch_y, batch_y_int, self.max_num_samples)
            sv_array = self.compute_sv_knn(model, sess, test_sv, train_sv)
            sv_array_mean = np.mean(sv_array, axis=0)
            n_train = train_sv[3]
            train_sample_indices = train_sv[4]
            return sv_array_mean, n_train, train_sample_indices

    def get_data_for_get_sv_mem(self, batch_x, batch_y, batch_y_int, num_samples):
        input_indices = np.arange(self.input_batch_size)
        test_sv = self.get_data_in_list(batch_x, batch_y, batch_y_int, input_indices)
        train_sv = self.get_stratified_sampling_data_for_sv(batch_x, batch_y, batch_y_int, num_samples, bool_adjust_balance=False)
        return test_sv, train_sv

    def get_data_for_update_sv_mem(self, batch_x, batch_y, batch_y_int, num_samples):
        # test set is solely from previous tasks when self.count_curr <= self.accum_size
        if (self.is_accum_curr) and (self.current_task_id != 0) and (self.count_curr <= self.accum_size):
            excl_indices = np.where(self.episodic_task == self.current_task_id)[0]
        else:
            excl_indices = ()
        unfilled_indices = np.where(self.episodic_labels_int == -1)[0]
        excl_indices = np.concatenate((excl_indices, unfilled_indices), axis=0)
        test_sv = self.get_stratified_sampling_data_for_sv(batch_x, batch_y, batch_y_int, num_samples, excl_indices=excl_indices)
        excl_indices = np.concatenate((test_sv[4], excl_indices), axis=0)

        train_sv = self.get_simple_random_sampling_data_for_sv(batch_x, batch_y, batch_y_int, excl_indices, num_samples)
        return test_sv, train_sv

    def get_stratified_sampling_data_for_sv(self, batch_x, batch_y, batch_y_int, num_samples, bool_adjust_balance=True, excl_indices = ()):
        sample_indices = self.get_stratified_sample_indices_for_sv(num_samples, excl_indices)
        x = self.episodic_images[sample_indices]
        y_ = self.episodic_labels[sample_indices]
        y_int = self.episodic_labels_int[sample_indices]
        if bool_adjust_balance and not self.is_accum_curr:
            x, y_, y_int = self.adjust_class_balance(x, y_, y_int, batch_x, batch_y, batch_y_int)

        stratified_sampling_data = self.get_data_in_list(x, y_, y_int, sample_indices)
        return stratified_sampling_data

    def get_data_in_list(self, x, y_, y_int, indices):
        num_data = y_int.shape[0]
        data_in_list = [x, y_, y_int, num_data, indices]
        return data_in_list

    def get_stratified_sample_indices_for_sv(self, num_samples, excl_indices):
        strat_sample_indices = np.array([])
        num_strat_sample = num_samples // self.out_dim
        for c in self.class_range:
            c_ind = np.nonzero(self.episodic_labels_int == c)[0]
            c_ind = np.setdiff1d(c_ind, excl_indices)
            c_ind = self.rng.permutation(c_ind)[0:num_strat_sample]
            strat_sample_indices = np.concatenate((strat_sample_indices, c_ind), axis=0)
        return strat_sample_indices.astype(int)

    def adjust_class_balance(self, x, y_, y_int, batch_x, batch_y, batch_y_int):
        lab_prop, expected_lab_prop = self.get_buffer_class_proportion()
        rand_threshold_lab_prop = self.rng.uniform(0, expected_lab_prop)
        idx_input_examples_with_rare_labels = np.nonzero(lab_prop[batch_y_int] < rand_threshold_lab_prop)
        rare_batch_x = batch_x[idx_input_examples_with_rare_labels]
        rare_batch_y = batch_y[idx_input_examples_with_rare_labels]
        rare_batch_y_int = batch_y_int[idx_input_examples_with_rare_labels]
        x = np.concatenate((x, rare_batch_x), axis=0)
        y_ = np.concatenate((y_, rare_batch_y), axis=0)
        y_int = np.concatenate((y_int, rare_batch_y_int), axis=0)
        return x, y_, y_int

    def get_buffer_class_proportion(self):
        curr_labels = self.episodic_labels[self.episodic_labels_int > -1]
        curr_labels_num = np.sum(curr_labels, axis=0)
        curr_labels_proportion = curr_labels_num / np.sum(curr_labels_num)
        expected_min_proportion_each_class = 1 / self.out_dim
        return curr_labels_proportion, expected_min_proportion_each_class

    def get_simple_random_sampling_data_for_sv(self, batch_x, batch_y, batch_y_int, excluded_indices, num_samples):
        sample_indices = self.get_simple_random_sample_indices_for_sv(excluded_indices, num_samples, batch_y_int)
        x = self.episodic_images[sample_indices]
        y_ = self.episodic_labels[sample_indices]
        y_int = self.episodic_labels_int[sample_indices]

        # concatenate input batch samples to the train set
        if (self.is_accum_curr) and (self.current_task_id != 0) and (self.count_curr <= self.accum_size):
            simple_rand_sampling_data = self.get_data_in_list(x, y_, y_int, sample_indices)
        else:
            x = np.concatenate((x, batch_x), axis=0)
            y_ = np.concatenate((y_, batch_y), axis=0)
            y_int = np.concatenate((y_int, batch_y_int), axis=0)
            simple_rand_sampling_data = self.get_data_in_list(x, y_, y_int, sample_indices)
        return simple_rand_sampling_data

    def get_simple_random_sample_indices_for_sv(self, excluded_indices, num_samples, batch_y_int):
        all_indices = np.arange(self.episodic_mem_size)
        sample_indices_array = np.setdiff1d(all_indices, excluded_indices)
        if self.is_penal:
            # num_samples = c + b + s;
            num_prev = np.setdiff1d(self.episodic_labels_int, batch_y_int).shape[0]
            num_s = (num_samples // self.out_dim) * num_prev
            num_c = (num_samples // self.out_dim) * (max(num_prev, 2))

            c_mask = np.isin(self.episodic_labels_int, batch_y_int)
            c_mask_ind = np.nonzero(c_mask)[0]
            c_mask_ind = np.setdiff1d(c_mask_ind, excluded_indices)
            s_mask_ind = np.setdiff1d(sample_indices_array, c_mask_ind)

            c_sample_indices = self.rng.permutation(c_mask_ind)[0:num_c]
            sample_indices = self.rng.permutation(s_mask_ind)[0:num_s]
            selected_sample_indices = np.concatenate((sample_indices, c_sample_indices), axis=0)
        else:
            selected_sample_indices = self.rng.permutation(sample_indices_array)[0:num_samples]
        return selected_sample_indices

    def compute_sv_knn(self, model, sess, test_sv, train_sv):
        k = self.num_k
        n_test = test_sv[3]
        n_train = train_sv[3]
        sv_array = np.ones((n_test, n_train)) * -1

        df_test, df_train = self.get_deep_features(model, sess, test_sv, train_sv)

        sorted_idx_mat = self.get_sorted_indices(df_test, df_train, train_sv)
        test_labels_int = test_sv[2]
        train_labels_int = train_sv[2]
        row_idx = np.arange(n_test)

        t_l = test_labels_int # test labels
        s_t_l = train_labels_int[sorted_idx_mat]  # sorted train labels
        sv_array[row_idx, sorted_idx_mat[:, n_train - 1]] = self.ind_f(s_t_l[:, n_train - 1], t_l) / n_train

        for i in reversed(range(n_train - 1)):
            sv_array[row_idx, sorted_idx_mat[:, i]] = \
                sv_array[row_idx, sorted_idx_mat[:, i + 1]] \
                + (self.ind_f(s_t_l[:, i], t_l) - self.ind_f(s_t_l[:, i + 1], t_l)) * min(k, i + 1) / (k * (i + 1))

        # sv_array_mean = np.mean(sv_array, axis=0)
        return sv_array

    def get_deep_features(self, model, sess, test_sv, train_sv):
        num = test_sv[0].shape[0] + train_sv[0].shape[0]
        total_x = np.concatenate((test_sv[0], train_sv[0]), axis=0)
        total_y_ = np.concatenate((test_sv[1], train_sv[1]), axis=0)

        # compute deep features with mini-batches
        bs = 64
        num_itr = num // bs + int(num % bs > 0)
        sid = 0
        deep_features = []
        for i in range(num_itr):
            eid = sid + bs if i != num_itr - 1 else num
            batch_x, batch_y = total_x[sid: eid], total_y_[sid: eid]
            feed_dict = {model.x: batch_x, model.y_: batch_y}
            # if 'resnet' in model.arch:
            feed_dict.update({model.train_phase: False})
            batch_deep_features = sess.run(model.features, feed_dict=feed_dict)
            deep_features.append(batch_deep_features)
            sid = eid
        if num_itr == 1:
            deep_features = deep_features[0]
        else:
            deep_features = np.concatenate(deep_features, axis=0)
        df_test = deep_features[0:test_sv[3]]
        df_train = deep_features[test_sv[3]:]
        return df_test, df_train

    def get_sorted_indices(self, df_test, df_train, train_sv):
        n_test = df_test.shape[0]
        n_train = train_sv[3]
        df_test_repeat = np.repeat(df_test, n_train, axis=0)
        df_train_tile = np.tile(df_train, (n_test, 1))
        distance_metric = self.get_distance_metric(df_test_repeat, df_train_tile)
        distance_metric_partition = np.reshape(distance_metric, (n_test, n_train))
        sorted_idx = np.argsort(distance_metric_partition, kind='mergesort', axis=1)
        return sorted_idx

    def get_distance_metric(self, df_test, df_train):
        # dist -> 0: unnormalized Euclidean distance
        # dist -> 1: normalized Euclidean distance
        # dist -> 2: cosine similarity
        if self.dist in [1, 2]:
            df_test = self.get_row_normalized_matrix(df_test)
            df_train = self.get_row_normalized_matrix(df_train)
        if self.dist in [0, 1]:
            distance_metric = self.get_euclidean_distance(df_test, df_train)
        elif self.dist == 2:
            distance_metric = self.get_cos_sim(df_test, df_train) * -1
        return distance_metric

    def get_euclidean_distance(self, df_test, df_train):
        euclidean_distance = np.sqrt(np.sum(((df_test - df_train) ** 2), axis=1))
        return euclidean_distance

    def get_cos_sim(self, df_test, df_train):
        cos_sim = np.sum(np.multiply(df_test, df_train), axis=1)
        return cos_sim

    def get_row_normalized_matrix(self, matrix):
        denom = np.reshape(np.sqrt(np.sum((matrix ** 2), axis=1)), (-1, 1))
        norm_matrix = np.divide(matrix, denom)
        return norm_matrix

    def ind_f(self, a1, a2):
        # indicator function (returns 1 if a1 == a2, 0 otherwise)
        return 1.0 * (a1 == a2)

class TempSVKNNBuffer(Buffer):
    def __init__(self, args, episodic_mem_size, in_dim, out_dim, eps_mem_batch):
        super(TempSVKNNBuffer, self).__init__(args, episodic_mem_size, in_dim, out_dim, eps_mem_batch)