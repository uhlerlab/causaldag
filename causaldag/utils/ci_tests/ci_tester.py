from typing import NewType, Callable, Dict, Union, List
import time
CI_Test = NewType('CI_Test', Callable[[Union[int, List[int]], Union[int, List[int]]], Dict])


class CI_Tester:
    def __init__(self):
        pass

    def is_ci(self, i, j, cond_set=set()):
        raise NotImplementedError


class MemoizedCI_Tester(CI_Tester):
    def __init__(self, ci_test: CI_Test, suffstat: Dict, track_times=False, detailed=False, **kwargs):
        CI_Tester.__init__(self)
        self.ci_dict_detailed = dict()
        self.ci_dict = dict()
        self.ci_test = ci_test
        self.suffstat = suffstat
        self.kwargs = kwargs
        self.detailed = detailed
        self.track_times = track_times
        self.ci_times = dict()

    def is_ci(self, i, j, cond_set=set()):
        i_, j_ = sorted((i, j))
        index = (i_, j_, frozenset(cond_set))

        # check if result exists and return
        _is_ci = self.ci_dict.get(index)
        if _is_ci is not None:
            return _is_ci

        # otherwise, compute result and save
        if self.track_times:
            start = time.time()
        test_results = self.ci_test(self.suffstat, i_, j_, cond_set=cond_set, **self.kwargs)
        if self.track_times:
            self.ci_times[index] = time.time() - start
        if self.detailed:
            self.ci_dict_detailed[index] = test_results
        _is_ci = not test_results['reject']
        self.ci_dict[index] = _is_ci

        return _is_ci


class PlainCI_Tester(CI_Tester):
    def __init__(self, ci_test: CI_Test, suffstat: Dict, **kwargs):
        CI_Tester.__init__(self)
        self.ci_test = ci_test
        self.suffstat = suffstat
        self.kwargs = kwargs
        
    def is_ci(self, i, j, cond_set=set()):
        return self.ci_test(self.suffstat, i, j, cond_set=cond_set, **self.kwargs)
        

