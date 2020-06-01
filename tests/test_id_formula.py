from unittest import TestCase
import unittest
from causaldag import IDFormula, MarginalDistribution, Product, ProbabilityTerm


class TestIDFormula(TestCase):
    def test_marginalizing_probability_term(self):
        base = ProbabilityTerm({'x1', 'x2', 'x3'})
        p_x1 = base.get_conditional(marginal={'x1'})
        print(p_x1)
        self.assertEqual(p_x1.active_variables, {'x1'})

        p_x12 = base.get_conditional(marginal={'x1', 'x2'})
        print(p_x12)
        self.assertEqual(p_x12.active_variables, {'x1', 'x2'})

    def test_marginalizing_prob_term_twice(self):
        base = ProbabilityTerm({'x1', 'x2', 'x3'})
        p = base.get_conditional(marginal={'x1', 'x2'})
        p2 = p.get_conditional(marginal={'x1'})
        self.assertEqual(p2.active_variables, {'x1'})
        self.assertEqual(p2.cond_set, set())
        self.assertEqual(p2.intervened, set())

        base = ProbabilityTerm({'x1', 'x2', 'x3'}, cond_set={'x4'})
        p = base.get_conditional(marginal={'x1', 'x2'})
        p2 = p.get_conditional(marginal={'x1'})
        self.assertEqual(p2.active_variables, {'x1'})
        self.assertEqual(p2.cond_set, {'x4'})
        self.assertEqual(p2.intervened, set())

        base = ProbabilityTerm({'x1', 'x2', 'x3'}, intervened={'x4'})
        p = base.get_conditional(marginal={'x1', 'x2'})
        p2 = p.get_conditional(marginal={'x1'})
        self.assertEqual(p2.active_variables, {'x1'})
        self.assertEqual(p2.cond_set, set())
        self.assertEqual(p2.intervened, {'x4'})

    def test_conditioning_probability_term(self):
        base = ProbabilityTerm({'x1', 'x2', 'x3'})
        p_x1 = base.get_conditional(marginal={'x1'}, cond_set={'x2', 'x3'})
        print(p_x1)
        self.assertEqual(p_x1.active_variables, {'x1'})

        p_x12 = base.get_conditional(marginal={'x1', 'x2'}, cond_set={'x3'})
        print(p_x12)
        self.assertEqual(p_x12.active_variables, {'x1', 'x2'})

        p_x1 = p_x12.get_conditional(marginal={'x1'}, cond_set={'x2'})
        print(p_x1)
        self.assertEqual(p_x1.active_variables, {'x1'})

    def test_marginalizing_product(self):
        base = Product([
            ProbabilityTerm({'x1', 'x2'}, {'x3'}),
            ProbabilityTerm({'x1', 'x2'}, {'x4'})
        ])
        p = base.get_conditional(marginal={'x1'})
        print(p)
        self.assertEqual(p.active_variables, {'x1'})

    def test_conditioning_product(self):
        base = Product([
            ProbabilityTerm({'x1', 'x2'}, {'x3'}),
            ProbabilityTerm({'x1', 'x2'}, {'x4'})
        ])
        p = base.get_conditional(marginal={'x1'}, cond_set={'x2'})
        print(p)
        self.assertEqual(p.active_variables, {'x1'})

    def test_marginalizing_marginal(self):
        base = Product([
            ProbabilityTerm({'x1', 'x2', 'x3'}, {'x4'}),
            ProbabilityTerm({'x1', 'x2', 'x3'}, {'x5'})
        ])
        p = base.get_conditional(marginal={'x1', 'x2'})
        p2 = p.get_conditional(marginal={'x1'})
        self.assertEqual(p2.active_variables, {'x1'})

        base = Product([
            ProbabilityTerm({'x1', 'x2', 'x3'}, intervened={'x4'}),
            ProbabilityTerm({'x1', 'x2', 'x3'}, intervened={'x5'})
        ])
        p = base.get_conditional(marginal={'x1', 'x2'})
        p2 = p.get_conditional(marginal={'x1'})
        self.assertEqual(p2.active_variables, {'x1'})
        print(p2)

    def test_conditioning_marginal(self):
        base = Product([
            ProbabilityTerm({'x1', 'x2', 'x3'}, {'x4'}),
            ProbabilityTerm({'x1', 'x2', 'x3'}, {'x5'})
        ])
        m = base.get_conditional(marginal={'x1', 'x2'})
        p = m.get_conditional(marginal={'x1'}, cond_set={'x2'})
        self.assertEqual(p.active_variables, {'x1'})
        self.assertEqual(p.marginalized_variables, {'x3'})
        print(p)


if __name__ == '__main__':
    unittest.main()
