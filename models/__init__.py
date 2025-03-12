from .TTD_l2p import TTD_Model as TTD_L2P_known_K


model_dict = {
    'TTD_L2P_known_K': TTD_L2P_known_K,
    'None': None,
}

def get_model(args):
    return {
        'ttd_model': model_dict[args.get('ttd_model', 'None')],
    }