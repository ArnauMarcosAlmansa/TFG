import time

import numpy as np

from src.config import device
import torch as t


class PositionalEncode(t.nn.Module):
    def __init__(self, L, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.L = L

    def forward(self, x):
        return self.do_positional_encoding(x)

    def do_positional_encoding(self, inputs):
        result = t.zeros(inputs.shape[0], inputs.shape[1] * self.L * 2, device=device)
        for i in range(inputs.shape[1]):
            for l in range(self.L):
                result[:, i * self.L * 2 + l * 2] = t.sin(2 ** l * np.pi * inputs[:, i])
                result[:, i * self.L * 2 + l * 2 + 1] = t.cos(2 ** l * np.pi * inputs[:, i])

        return result

    def do_positional_encoding2(self, inputs):
        result = t.zeros(inputs.shape[0], inputs.shape[1] * self.L * 2, device=device)
        xp = np.pi * inputs[:, 0]
        yp = np.pi * inputs[:, 1]
        zp = np.pi * inputs[:, 2]

        sxp = t.sin(xp)
        cxp = t.cos(xp)
        syp = t.sin(yp)
        cyp = t.cos(yp)
        szp = t.sin(zp)
        czp = t.cos(zp)
        n = 20

        result[:, 0    ] = sxp
        result[:, 1    ] = cxp
        result[:, 0+n  ] = syp
        result[:, 1+n  ] = cyp
        result[:, 0+n+n] = szp
        result[:, 1+n+n] = czp
        for k in range(1, self.L):
            k2 = 2*k
            result[:, k2      ] = 2*result[:, k2-2    ]*result[:, k2-1    ]
            result[:, k2+1    ] = result[:, k2-1    ]*result[:, k2-1    ] - result[:, k2-2    ]*result[:, k2-2    ]
            result[:, k2  +n  ] = 2*result[:, k2-2+n  ]*result[:, k2-1+n  ]
            result[:, k2+1+n  ] = result[:, k2-1+n  ]*result[:, k2-1+n  ] - result[:, k2-2+n  ]*result[:, k2-2+n  ]
            result[:, k2  +n+n] = 2*result[:, k2-2+n+n]*result[:, k2-1+n+n]
            result[:, k2+1+n+n] = result[:, k2-1+n+n]*result[:, k2-1+n+n] - result[:, k2-2+n+n]*result[:, k2-2+n+n]
        return result


if __name__ == '__main__':
    import cProfile

    pe = PositionalEncode(10)

    data = t.rand((120, 3))

    cProfile.run('pe.do_positional_encoding2(data)')
    #
    # start = time.perf_counter()
    # res1 = pe.do_positional_encoding(data)
    #
    # mid = time.perf_counter()
    # res2 = pe.do_positional_encoding2(data)
    #
    # end = time.perf_counter()

    # print(mid - start)
    # print(end - mid)
    print()
    