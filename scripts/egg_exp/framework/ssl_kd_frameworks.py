import torch
from .interface import Framework

class SSL_KD_Framework(Framework):
    KD_MODE = 0
    FT_MODE = 1
    
    def __init__(self, teacher, student, backend, kd_loss, ft_loss):
        super(SSL_KD_Framework, self).__init__()
        self.add_module('teacher', teacher, flag_train=False)
        self.add_module('student', student, flag_train=True)
        self.add_module('backend', backend, flag_train=True)
        self.add_module('kd_loss', kd_loss, flag_train=True)
        self.add_module('ft_loss', ft_loss, flag_train=True)
        
    def __call__(self, x, x_teacher=None, kd_label=None, ft_label=None):
        # student
        x = self.modules['student'](x)
        
        # loss (KD)
        kd_loss = None
        if x_teacher is not None:
            with torch.set_grad_enabled(False):
                kd_label = self.modules['teacher'](x_teacher, output_hidden_states=True).hidden_states
                kd_label = torch.stack(kd_label, dim=1)
        if kd_label is not None:
            kd_loss = self.modules['kd_loss'](x, kd_label)
        
        if self.mode == self.KD_MODE:
            return kd_loss
        
        # loss (FT)
        ft_loss = None
        x = self.modules['backend'](x)
        if ft_label is not None:
            ft_loss = self.modules['ft_loss'](x, ft_label)
            return x, ft_loss
        else:
            return x
        
    def set_kd_mode(self):
        self.mode = self.KD_MODE
        self.set_module_trainability('student', True)
        self.set_module_trainability('kd_loss', True)
        self.set_module_trainability('backend', False)
        self.set_module_trainability('ft_loss', False)
            
    def set_ft_mode(self, freeze_student=True):
        self.mode = self.FT_MODE
        self.set_module_trainability('student', freeze_student)
        self.set_module_trainability('kd_loss', False)
        self.set_module_trainability('backend', True)
        self.set_module_trainability('ft_loss', True)
            
class SSL_TAKD_Framework(Framework):
    def __init__(self, teacher, student, backend, kd_loss, ft_loss, use_ft_adapter=False):
        super(SSL_TAKD_Framework, self).__init__()
        self.add_module('teacher', teacher, flag_train=False)
        self.add_module('student', student, flag_train=True)
        self.add_module('backend', backend, flag_train=True)
        self.add_module('kd_loss', kd_loss, flag_train=True)
        self.add_module('ft_loss', ft_loss, flag_train=True)
        self.use_ft_adapter = use_ft_adapter
        
    def __call__(self, x_ft, x_kd=None, x_teacher=None, kd_label=None, ft_label=None):
        # student
        if x_kd is not None:
            batch_kd = x_kd.size(0)
            x = torch.cat((x_kd, x_ft), dim=0)
            idx_without_adapter = batch_kd if self.use_ft_adapter else None
            x = self.modules['student'](x, idx_without_adapter=idx_without_adapter)
            x_kd, x_ft = x[:batch_kd], x[batch_kd:]
        else:
            x_ft = self.modules['student'](x_ft)
        
        # loss (KD)
        kd_loss = None
        if x_teacher is not None or kd_label is not None:
            if x_teacher is not None:
                with torch.set_grad_enabled(False):
                    kd_label = self.modules['teacher'](x_teacher, output_hidden_states=True).hidden_states
                    kd_label = torch.stack(kd_label, dim=1)
            kd_loss = self.modules['kd_loss'](x_kd, kd_label)
        
        # backend
        x_ft = self.modules['backend'](x_ft)
        
        # loss (FT)
        ft_loss = None
        if ft_label is not None:
            ft_loss = self.modules['ft_loss'](x_ft, ft_label)
            
        if kd_loss is None and ft_loss is None:
            return x_ft
        else:
            return x_ft, kd_loss, ft_loss