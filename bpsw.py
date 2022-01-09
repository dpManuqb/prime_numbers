from typing import Union, List, Tuple
from millerrabin import MillerRabin
from lucas import Lucas

############################################################################
################################ Main class ################################
############################################################################

class BPSW:

    @staticmethod
    def test(n:int, a:Union[int,List[int]]=2, P_Q:Union[Tuple[int,int],List[Tuple[int,int]]]=None):
        guess = MillerRabin.test(n, a)

        if(guess == False):
            return False

        return Lucas.strong_test(n, P_Q)

    @staticmethod
    def multiprocess_test(n:int, a:Union[int,List[int]]=2, P_Q:Union[Tuple[int,int],List[Tuple[int,int]]]=None, processes:int=2):
        guess = MillerRabin.multiprocess_test(n, a, processes)

        if(guess == False):
            return False

        return Lucas.multiprocess_strong_test(n, P_Q, processes)