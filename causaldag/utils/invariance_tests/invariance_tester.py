from typing import NewType, Callable, Dict, Any, Set
import time

InvarianceTest = NewType('InvarianceTest', Callable[[Dict, Any, Any, Set], Dict])


class InvarianceTester:
    def __init__(self):
        pass

    def is_invariant(self, node, context, cond_set=set()):
        raise NotImplementedError


class MemoizedInvarianceTester(InvarianceTester):
    def __init__(self, invariance_test: InvarianceTest, suffstat: Dict, track_times=False, detailed=False, **kwargs):
        """
        Class for memoizing the results of invariance tests.

        Parameters
        ----------
        invariance_test:
            Function taking suffstat, context, node, and conditioning set, and returning a dictionary that includes
            the key 'reject'.
        suffstat:
            Dictionary containing sufficient statistics for all contexts.
        track_times:
            If True, keep a dictionary mapping each invariance test to the time taken to perform it.
        detailed:
            If True, keep a dictionary mapping each invariance test to its full set of results.
        **kwargs:
            Additional keyword arguments to be passed to the invariance test.

        See Also
        --------
        PlainInvarianceTester

        Example
        -------
        """
        InvarianceTester.__init__(self)
        self.invariance_dict_detailed = dict()
        self.invariance_dict = dict()
        self.invariance_test = invariance_test
        self.suffstat = suffstat
        self.kwargs = kwargs
        self.detailed = detailed
        self.track_times = track_times
        self.invariance_times = dict()

    def is_invariant(self, node, context, cond_set=set()):
        """
        Check if the conditional distribution of node, given cond_set, is invariant to the context.
        """
        index = (node, context, frozenset(cond_set))

        # check if result exists and return
        _is_invariant = self.invariance_dict.get(index)
        if _is_invariant is not None:
            return _is_invariant

        # otherwise, compute result and save
        if self.track_times:
            start = time.time()
        test_results = self.invariance_test(
            self.suffstat,
            context,
            node,
            cond_set=cond_set,
            **self.kwargs
        )
        if self.track_times:
            self.invariance_times[index] = time.time() - start
        if self.detailed:
            self.invariance_dict_detailed[index] = test_results
        _is_invariant = not test_results['reject']
        self.invariance_dict[index] = _is_invariant

        return _is_invariant


class PlainInvarianceTester(InvarianceTester):
    def __init__(self, invariance_test: InvarianceTest, suffstat: Dict, **kwargs):
        """
        Class for returning the results of invariance tests.

        Parameters
        ----------
        invariance_test:
            Function taking suffstat, context, node, and conditioning set, and returning a dictionary that includes
            the key 'reject'.
        suffstat:
            Dictionary containing sufficient statistics for all contexts.
        **kwargs:
            Additional keyword arguments to be passed to the invariance test.

        See Also
        --------
        MemoizedInvarianceTester

        Example
        -------

        """
        InvarianceTester.__init__(self)
        self.invariance_test = invariance_test
        self.suffstat = suffstat
        self.kwargs = kwargs

    def is_invariant(self, node, context, cond_set=set()):
        return self.invariance_test(
            self.suffstat,
            context,
            node,
            cond_set=cond_set,
            **self.kwargs
        )

