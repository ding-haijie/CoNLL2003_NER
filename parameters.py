params = {
    'batch_size': 256,
    'max_epoch': 50,
    'feature_size': 6,  # casing features
    'feature_dim': 10,  # casing features hidden dim
    'word_embed_dim': 100,
    'char_embed_dim': 40,
    'hidden_dim': 256,
    'dropout_p': 0.3,
    'optimizer': 'sgd',
    'learning_rate': 0.001,
    'momentum': 0.9,
    'weight_decay': 0.0,
    'grad_clip': 10.0,
    'device_id': 1,
    'seed': 16,
    'train_mode': True,
    'resume': False,
}
