from collections import Counter

import numpy as np

from utils.global_vars import RANDOM_MEM_RETRIEVE

class Buffer:
    def __init__(self, args, episodic_mem_size, in_dim, out_dim, eps_mem_batch):
        self.args = args
        # self.task_labels = task_labels
        self.episodic_images = np.zeros([episodic_mem_size, 32, 32, 3])
        self.episodic_labels = np.zeros([episodic_mem_size, out_dim])
        self.episodic_task = np.zeros([episodic_mem_size])
        self.episodic_task.fill(-1)
        self.episodic_mem_size = episodic_mem_size
        self.count_cls = np.zeros(out_dim, dtype=np.int32)
        self.eps_mem_batch = eps_mem_batch
        self.examples_seen_so_far = 0
        self.rng = np.random.RandomState(args.random_seed)
        self.episodic_labels_int = np.zeros(self.episodic_mem_size, dtype=int) - 1

    def get_mem(self, model, sess, task, current_x=None, current_y=None, exclude=None):
        self.task_id = task

        mem_filled_so_far = self.examples_seen_so_far if (
                self.examples_seen_so_far < self.episodic_mem_size) else self.episodic_mem_size

        if mem_filled_so_far < self.eps_mem_batch:
            er_mem_indices = np.arange(mem_filled_so_far)
            self.rng.shuffle(er_mem_indices)
            final_x, final_y = self.episodic_images[er_mem_indices], self.episodic_labels[er_mem_indices]
        else:
            # if self.args.model in RANDOM_MEM_RETRIEVE and not self.args.is_mir:
            #     er_mem_indices = self.rng.choice(mem_filled_so_far, self.eps_mem_batch, replace=False)
            #     self.rng.shuffle(er_mem_indices)
            #     final_x, final_y = self.episodic_images[er_mem_indices], self.episodic_labels[er_mem_indices]
            # elif self.args.model == 'MIR' or self.args.is_mir:
                #assert exclude is not None, "current task id should be passed"
                # retrieve some samples to do MIR
            subsample = 50
            if subsample <= self.eps_mem_batch:
                raise Exception('subsample need to be larger than eps_mem_batch')
            valid_idx = np.where(self.episodic_task != exclude)[0]
            valid_idx = valid_idx[valid_idx < mem_filled_so_far]
            # if self.args.balanced_sampling:
            #     subsample_idx = np.empty((0, ), dtype=np.int32)
            #     classes = np.unique(self.episodic_labels[valid_idx].argmax(axis=1))
            #     subsample_per_class = subsample // len(classes)
            #     for c in classes:
            #         valid_idx_c = valid_idx[self.episodic_labels[valid_idx].argmax(axis=1) == c]
            #         subsample_idx = np.hstack((subsample_idx, self.rng.choice(valid_idx_c, subsample_per_class, replace=False)))
            # else:
            subsample_idx = self.rng.choice(valid_idx, subsample, replace=False)
            subsample_x, subsample_y = self.episodic_images[subsample_idx], self.episodic_labels[subsample_idx]
            feed_dict_subsample = {model.x: subsample_x, model.y_: subsample_y, model.flag1: 0, model.keep_prob: 1.0}
            # if 'resnet' in self.args.arch:
            feed_dict_subsample.update({model.train_phase: False})
            nd_logit_mask = np.zeros([17, 100])
            nd_logit_mask[:] = 0
            for tt in range(self.task_id + 1):
                nd_logit_mask[tt][self.task_labels[tt]] = 1.0
            logit_mask_dict = {m_t: i_t for (m_t, i_t) in zip(model.output_mask, nd_logit_mask)}
            feed_dict_subsample.update(logit_mask_dict)
            feed_dict_subsample[model.mem_batch_size] = float(subsample_x.shape[0])
            loss_pre = sess.run(model.cross_entropy, feed_dict=feed_dict_subsample)

            sess.run(model.set_star_vars)
            # if model.args.optimizer == 'Adam':
            #     sess.run(model.adam_save)
            feed_dict_current = {model.x: current_x, model.y_: current_y, model.flag1: 0, model.keep_prob: 1.0}
            feed_dict_current.update(logit_mask_dict)
            feed_dict_current[model.mem_batch_size] = float(current_x.shape[0])
            # if 'resnet' in self.args.arch:
            feed_dict_current.update({model.train_phase: True})
            # sess.run(model.train, feed_dict=feed_dict_current)
            loss_post = sess.run(model.cross_entropy, feed_dict=feed_dict_subsample)
            sess.run(model.restore_weights)
            # if model.args.optimizer == 'Adam':
            #     sess.run(model.adam_restore)
            scores = loss_post - loss_pre
            idx_in_subsample = scores.argsort()[::-1][:self.eps_mem_batch]
            self.rng.shuffle(idx_in_subsample)
            final_x, final_y = subsample_x[idx_in_subsample], subsample_y[idx_in_subsample]

        return final_x, final_y

    def update_mem(self, batch_x, batch_y, task_id):
        batch_y_int = np.argmax(batch_y, axis=1)
        for er_x, er_y, er_y_int in zip(batch_x, batch_y, batch_y_int):
            if self.episodic_mem_size > self.examples_seen_so_far:
                self.episodic_images[self.examples_seen_so_far] = er_x
                self.episodic_labels[self.examples_seen_so_far] = er_y
                self.episodic_labels_int[self.examples_seen_so_far] = er_y_int
                self.episodic_task[self.examples_seen_so_far] = task_id
            else:
                j = self.rng.randint(0, self.examples_seen_so_far)
                if j < self.episodic_mem_size:
                    self.episodic_images[j] = er_x
                    self.episodic_labels[j] = er_y
                    self.episodic_task[j] = task_id
                    self.episodic_labels_int[j] = er_y_int
            self.examples_seen_so_far += 1

    def show_buffer_status(self):
        return Counter(self.episodic_task.tolist())


class GSS_Buffer(Buffer):
    def __init__(self, args, episodic_mem_size, in_dim, out_dim, eps_mem_batch, task_labels):
        super(GSS_Buffer, self).__init__(args, episodic_mem_size, in_dim, out_dim, eps_mem_batch, task_labels)
        self.task_labels = task_labels
        self.max_num_sample_grad = 10
        self.episodic_mem_score = np.zeros([episodic_mem_size])
        self.rng = np.random.RandomState(args.random_seed)

    def get_mem(self, model, sess, current_x, current_y, iter):
        mem_filled_so_far = self.examples_seen_so_far if (
                self.examples_seen_so_far < self.episodic_mem_size) else self.episodic_mem_size
        eff_mem_batch = min(self.eps_mem_batch, mem_filled_so_far)
        if iter == 0:
            self.b_ind = 0
            er_mem_indices = np.arange(0, mem_filled_so_far)
            self.rng.shuffle(er_mem_indices)
            self.er_mem_indices = er_mem_indices

        ind = self.er_mem_indices[self.b_ind * eff_mem_batch: (self.b_ind + 1) * eff_mem_batch]
        final_x, final_y = self.episodic_images[ind], self.episodic_labels[ind]
        self.b_ind += 1
        if (self.b_ind) * eff_mem_batch >= mem_filled_so_far:
            self.b_ind = 0
        return final_x, final_y

    def update_mem(self, sess, model, batch_x, batch_y, task_id):
        self.task_id = task_id
        mem_filled_so_far = self.examples_seen_so_far if (
                self.examples_seen_so_far < self.episodic_mem_size) else self.episodic_mem_size
        eff_mem_batch = min(self.eps_mem_batch, mem_filled_so_far)

        # find sample batch gradient vectors (i.e., one graident vector per sample batch)
        if (mem_filled_so_far > 0):
            sample_ind = np.arange(0, mem_filled_so_far)
            self.rng.shuffle(sample_ind)
            # find total number of sample batches -> num_sample_set = number of batch gradient vectors
            num_sample_set = min(self.max_num_sample_grad, mem_filled_so_far // eff_mem_batch)
            s_img_g = [];
            s_label_g = []
            for s in range(num_sample_set):
                s_ind_g = sample_ind[s * eff_mem_batch: (s + 1) * eff_mem_batch]
                s_img_g.append(self.episodic_images[s_ind_g])
                s_label_g.append(self.episodic_labels[s_ind_g])
            self.sample_grad = self.get_grad_vec(sess, model, s_img_g, s_label_g)

        # find gradient vectors of data in new input batch -> single gradient vector per each image in new input batch
        self.new_grad = self.get_grad_vec(sess, model, batch_x, batch_y, single=True)

        # fill in episodic memory until it's full
        if (self.examples_seen_so_far  < 1100):
            for i, new_g in enumerate(self.new_grad):
                mem_idx = self.examples_seen_so_far
                self.episodic_images[mem_idx] = batch_x[i]
                self.episodic_labels[mem_idx] = batch_y[i]
                self.episodic_task[mem_idx] = task_id
                # find max cosine sim score
                if (mem_filled_so_far > 0):
                    max_cos_sim = self.maximal_cosine_sim(new_g, self.sample_grad)
                    self.episodic_mem_score[mem_idx] = max_cos_sim
                # if this is the first time the memory is updated,
                # assign an arbitrary cosine sim score for all data in the first input batch
                else:
                    self.episodic_mem_score[mem_idx] = 0.1
                self.examples_seen_so_far += 1

        # replacement - gradient based sampling
        else:
            self.examples_seen_so_far += len(batch_x)
            # find a single batch gradient vector for input batch
            self.batch_new_grad = self.get_grad_vec(sess, model, [batch_x], [batch_y])
            max_batch_cos_sim = self.get_batch_cosine_sim(self.batch_new_grad, self.sample_grad)
            if (max_batch_cos_sim < 0):
                buffer_sim = (self.episodic_mem_score - np.min(self.episodic_mem_score)) / (
                            np.max(self.episodic_mem_score) - np.min(self.episodic_mem_score) + 0.01)
                buffer_sim_norm = buffer_sim / np.sum(buffer_sim)

                # draw candidates for replacement
                buffer_idx = self.rng.choice(self.episodic_mem_size, size=len(batch_x), replace=False,
                                              p=buffer_sim_norm.tolist())

                batch_item_sim = self.get_each_batch_cosine_sim(self.new_grad, self.sample_grad)   # similarity of batch grad to sample grad
                scaled_batch_item_sim = np.expand_dims((batch_item_sim + 1) / 2, axis=1)
                buffer_repl_batch_sim = np.expand_dims((self.episodic_mem_score[buffer_idx] + 1) / 2, axis=1)

                # draw random numbers to decide replacement
                prob = np.concatenate((scaled_batch_item_sim, buffer_repl_batch_sim), axis=1)
                prob = prob / np.sum(prob, axis=1)[:, None]                                                  # (10, 2)

                # sample indices
                outcome = [self.rng.choice(np.arange(prob.shape[1]), size=1, p=prob[k, :], replace=False) for k in range(prob.shape[0])]

                # replace samples with outcome = 1
                added_index = np.arange(batch_item_sim.shape[0])
                sub_index = np.concatenate(outcome).astype(bool)

                self.episodic_images[buffer_idx[sub_index]] = batch_x[added_index[sub_index]]
                self.episodic_labels[buffer_idx[sub_index]] = batch_y[added_index[sub_index]]
                self.episodic_mem_score[buffer_idx[sub_index]] = batch_item_sim[added_index[sub_index]]
                self.episodic_task[buffer_idx[sub_index]] = task_id

    def get_grad_vec(self, sess, model, images, labels, single=False):
        for ii, (xx, yy) in enumerate(zip(images, labels)):
            if (single) or xx.ndim == 1:
                xx = [xx]
                yy = [yy]
            feed_dict = {model.x: xx, model.y_: yy, model.flag1: 0, model.keep_prob: 1.0, model.train_phase: False}
            nd_logit_mask = np.zeros([17, 100])
            nd_logit_mask[:] = 0
            for tt in range(self.task_id + 1):
                nd_logit_mask[tt][self.task_labels[tt]] = 1.0
            logit_mask_dict = {m_t: i_t for (m_t, i_t) in zip(model.output_mask, nd_logit_mask)}
            feed_dict.update(logit_mask_dict)
            feed_dict[model.mem_batch_size] = float(1)
            # if 'resnet' in model.args.arch:
            feed_dict.update({model.train_phase: False})

            grad_vec = sess.run([model.vectorized_gradients],
                                feed_dict=feed_dict)
            if ii == 0:
                grad_vec_arr = np.array(grad_vec)
            else:
                grad_vec_arr = np.concatenate((grad_vec_arr, np.array(grad_vec)), axis=0)
        return grad_vec_arr

    def cosine_sim(self, v, m):
        dot = v @ m.T
        mm = np.linalg.norm(m, ord=2, axis=1, keepdims=True)
        vv = np.linalg.norm(v, ord=2, keepdims=True)
        norm = vv @ mm.T
        if not np.any(dot):
            norm = np.ones(dot.shape)
        cos_sim = np.squeeze(dot / norm)
        return cos_sim

    def get_each_batch_cosine_sim(self, new_grad_vec, sample_grad_vec_arr):
        cosine_sim = np.zeros(new_grad_vec.shape[0])
        for i, grad_i in enumerate(new_grad_vec):
            cosine_sim[i] = np.max(self.cosine_sim(grad_i, sample_grad_vec_arr))
        return cosine_sim

    def get_batch_cosine_sim(self, batch_grad_vec, sample_grad_vec_arr):
        return np.max(self.cosine_sim(batch_grad_vec, sample_grad_vec_arr))

    def maximal_cosine_sim(self, new_grad_vec, sample_grad_vec_arr):
        if new_grad_vec.ndim == 1:
            new_grad_vec = np.expand_dims(new_grad_vec, axis=0)
            cosine_sim = np.max(self.cosine_sim(new_grad_vec, sample_grad_vec_arr))
        else:
            cosine_sim = np.zeros(new_grad_vec.shape[0])
            for i, batch_grad in enumerate(new_grad_vec):
                cosine_sim[i] = np.max(self.cosine_sim(batch_grad, sample_grad_vec_arr))
        return cosine_sim
