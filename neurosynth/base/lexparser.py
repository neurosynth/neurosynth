# emacs: -*- mode: python-mode; py-indent-offset: 2; tab-width: 2; indent-tabs-mode: nil -*-
# ex: set sts=2 ts=2 sw=2 et:

import ply.lex as lex
import ply.yacc as yacc

import logging

logger = logging.getLogger('neurosynth.lexparser')


class Lexer(object):

    tokens = (
        'FEATURE', 'FLOAT', 'ANDNOT', 'OR', 'AND', 'CONTRAST', 'LPAR', 'RPAR', 'LT', 'RT'
    )

    t_FEATURE = r'[a-z\_\-\*]+'
    t_ANDNOT = r'\&\~'
    t_AND = r'\&'
    t_OR = r'\|'
    # t_CONTRAST = r'\%'
    t_LPAR = r'\('
    t_RPAR = r'\)'
    t_LT = r'\<'
    t_RT = r'\>'

    t_ignore = ' \t'

    def __init__(self, dataset=None):
        self.dataset = dataset

    def t_FLOAT(self, t):
        r'[0-9\.]+'
        t.value = float(t.value)
        return t

    def build(self, **kwargs):
        self.lexer = lex.lex(module=self, **kwargs)

    def t_error(self, t):
        logger.error("Illegal character %s!" % t.value[0])
        t.lexer.skip(1)

    def test(self, data):

        self.lexer.input(data)
        while True:
            tok = self.lexer.token()
            if not tok:
                break      # No more input
            print tok


class Parser(object):

    def __init__(self, lexer, dataset, threshold=0.001, func='sum'):

        self.lexer = lexer
        self.dataset = dataset
        self.threshold = threshold
        self.func = func
        self.tokens = lexer.tokens

    def p_list_andnot(self, p):
        'list : list ANDNOT list'
        p[0] = {k: v for k, v in p[1].items() if k not in p[3]}

    def p_list_and(self, p):
        'list : list AND list'
        p[0] = {k: v for k, v in p[1].items() if k in p[3]}

    def p_list_or(self, p):
        'list : list OR list'
        p[1].update(p[3])
        p[0] = p[1]

    def p_list_lt(self, p):
        'list : list LT freq'
        p[0] = {k: v for k, v in p[1].items() if v < p[3]}

    def p_list_rt(self, p):
        'list : list RT freq'
        p[0] = {k: v for k, v in p[1].items() if v >= p[3]}

    def p_list_feature(self, p):
        'list : FEATURE'
        p[0] = self.dataset.get_ids_by_features(p[
                                                1], self.threshold, self.func, get_weights=True)

    def p_list_expr(self, p):
        'list : LPAR list RPAR'
        p[0] = p[2]

    def p_freq_float(self, p):
        'freq : FLOAT'
        p[0] = p[1]

    def build(self, **kwargs):
        self.parser = yacc.yacc(module=self, **kwargs)

    def parse(self, input):
        return self.parser.parse(input)
