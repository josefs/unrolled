import Prelude
import qualified Prelude as P
import qualified Data.List.Unrolled as U

import Criterion
import Criterion.Main

main = defaultMain [
        bench "standard sum/enum"  
                  $ \n -> P.sum (P.enumFromTo (0+n::Int) (10000+n))
       ,bench "unrolled sum/enum"  
                  $ \n -> U.sum (U.enumFromToInt (0+n::Int) (10000+n))
       ,bench "unrolled sum/enum alternative" 
                  $ \n -> U.sum (U.enumFromToInt2 (0+n::Int) (10000+n))


       -- These benchmarks are a bit flawed since the standard version
       -- benefits from fusion
       ,bench "standard sum/filter/enum"
                  $ \n -> P.sum (P.filter odd (P.enumFromTo (0+n::Int) (10000+n)))
       ,bench "unrolled sum/filter/enum"
                  $ \n -> U.sum (U.filter odd (U.enumFromTo (0+n::Int) (10000+n)))
       ,bench "standard sum/filter/enum realigning"
                  $ \n -> U.sum (U.filterR odd (U.enumFromTo (0+n::Int) (10000+n)))
       ]
