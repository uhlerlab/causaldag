from typing import NewType, Callable, Dict, List
import time

InvarianceTest = NewType('InvarianceTest', Callable[[Dict, Dict], Dict])


class InvarianceTester:
    def __init__(self):
        pass

    def is_invariant(self, node, setting1, setting2, cond_set=set()):
        raise NotImplementedError


class MemoizedInvarianceTester(InvarianceTester):
    def __init__(self, invariance_test: InvarianceTest, suffstats: List[Dict], track_times=False, detailed=False, **kwargs):
        """
        Class for memoizing the results of invariance tests.

        Parameters
        ----------
        invariance_test:
            the DAG to which the SHD of the skeleton will be computed.
        suffstats:
            list mapping settings to their sufficient statistics
        track_times:
            if True, keep a dictionary mapping each invariance test to the time taken to perform it.
        detailed:
            if True, keep a dictionary mappint each invariance test to its full set of results.

        """
        InvarianceTester.__init__(self)
        self.invariance_dict_detailed = dict()
        self.invariance_dict = dict()
        self.invariance_test = invariance_test
        self.suffstats = suffstats
        self.kwargs = kwargs
        self.detailed = detailed
        self.track_times = track_times
        self.invariance_times = dict()

    def is_invariant(self, node, setting1, setting2, cond_set=set()):
        """
        Check if the conditional distribution of node, given cond_set, is invariant between setting1 and setting2
        """
        setting1_, setting2_ = sorted((setting1, setting2))
        index = (node, setting1_, setting2_, frozenset(cond_set))

        # check if result exists and return
        _is_invariant = self.invariance_dict.get(index)
        if _is_invariant is not None:
            return _is_invariant

        # otherwise, compute result and save
        if self.track_times:
            start = time.time()
        test_results = self.invariance_test(
            self.suffstats[setting1],
            self.suffstats[setting2],
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
    def __init__(self, invariance_test: InvarianceTest, suffstats: List[Dict], **kwargs):
        """
        Class for returning the results of invariance tests.

        Parameters
        ----------
        invariance_test:
            the DAG to which the SHD of the skeleton will be computed.
        suffstats:
            list mapping settings to their sufficient statistics
        """
        InvarianceTester.__init__(self)
        self.invariance_test = invariance_test
        self.suffstats = suffstats
        self.kwargs = kwargs

    def is_invariant(self, node, setting1, setting2, cond_set=set()):
        return self.invariance_test(
            self.suffstats[setting1],
            self.suffstats[setting2],
            node,
            cond_set=cond_set,
            **self.kwargs
        )

