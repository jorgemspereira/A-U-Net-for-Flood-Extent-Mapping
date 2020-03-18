from keras import backend as K
from keras.optimizers import Optimizer
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops

class AdaMod(Optimizer):
    def __init__(self,
                 lr=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 beta_3=0.999,
                 epsilon=None,
                 decay=0.,
                 #amsgrad=False,
                 **kwargs):
        super(AdaMod, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.beta_3 = K.variable(beta_3, name='beta_3')
            self.decay = K.variable(decay, name='decay')
            #self.amsgrad = K.variable(amsgrad, name='amsgrad')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay


    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = []

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (  # pylint: disable=g-no-augmented-assignment
                    1. /
                    (1. +
                     self.decay * math_ops.cast(self.iterations, K.dtype(self.decay))))

        with ops.control_dependencies([state_ops.assign_add(self.iterations, 1)]):
            t = math_ops.cast(self.iterations, K.floatx())
        lr_t = lr * (
                K.sqrt(1. - math_ops.pow(self.beta_2, t)) /
                (1. - math_ops.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        ss = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        # if self.amsgrad is True:
        #     vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        # else:
        vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + ss + vhats

        for p, g, m, v, s, vhat in zip(params, grads, ms, vs, ss, vhats):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * math_ops.square(g)
            # if self.amsgrad is True:
            #     vhat_t = math_ops.maximum(vhat, v_t)
            #     miu_t = lr_t / (K.sqrt(vhat_t) + self.epsilon)
            #     p_t = p - miu_t * m_t
            #     self.updates.append(state_ops.assign(vhat, vhat_t))
            # else:
            miu_t = lr_t / (K.sqrt(v_t) + self.epsilon)
            s_t = self.beta_3 * s + (1 - self.beta_3) * miu_t
            miu_t_hat = math_ops.minimum(miu_t, s_t)
            p_t = p - miu_t_hat * m_t
            self.updates.append(state_ops.assign(m, m_t))
            self.updates.append(state_ops.assign(v, v_t))
            self.updates.append(state_ops.assign(s, s_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(state_ops.assign(p, new_p))
        return self.updates


    def get_config(self):
        config = {
            'lr': float(K.get_value(self.lr)),
            'beta_1': float(K.get_value(self.beta_1)),
            'beta_2': float(K.get_value(self.beta_2)),
            'beta_3': float(K.get_value(self.beta_3)),
            'decay': float(K.get_value(self.decay)),
            'epsilon': self.epsilon
            # 'amsgrad': self.amsgrad
        }
        base_config = super(AdaMod, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))