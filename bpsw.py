from typing import Union, List, Tuple
from fermat import Fermat
from lucas import Lucas

############################################################################
################################ Main class ################################
############################################################################

class BPSW:

    @staticmethod
    def test(n:int, a_list:Union[int,List[int]]=2, P_Q:Union[Tuple[int,int],List[Tuple[int,int]]]=None, processes:int=1) -> bool:
        """Combination of a Fermat test and a Lucas test"""
        guess = Fermat.test(n, a_list=a_list, processes=processes)

        if(guess == False):
            return False

        return Lucas.test(n, P_Q=P_Q, processes=processes)

    @staticmethod
    def strong_test(n:int, a_list:Union[int,List[int]]=2, P_Q:Union[Tuple[int,int],List[Tuple[int,int]]]=None, processes:int=1) -> bool:
        """Combination of a Fermat strong test (MillerRabin) and a Lucas Strong test"""
        guess = Fermat.strong_test(n, a_list=a_list, processes=processes)

        if(guess == False):
            return False

        return Lucas.strong_test(n, P_Q=P_Q, processes=processes)