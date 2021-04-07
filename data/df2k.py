import os
from data import srdata

class DF2K(srdata.SRData):
    def __init__(self, args, name='DF2K', train=True, benchmark=False,train_arch=False):
        data_range = [r.split('-') for r in args.data_range.split('/')]
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            elif train_arch:
                data_range = data_range[1]
            else:
                data_range = data_range[2]

        self.begin, self.end = list(map(lambda x: int(x), data_range))
        super(DF2K, self).__init__(
            args, name=name, train=train, benchmark=benchmark,train_arch=train_arch
        )

    def _scan(self):
        names_hr, names_lr = super(DF2K, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(DF2K, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, 'DF2K_HR')
        self.dir_lr = os.path.join(self.apath, 'DF2K_LR_bicubic')
        if self.input_large: self.dir_lr += 'L'
