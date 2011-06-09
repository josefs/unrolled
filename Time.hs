import Prelude
import qualified Prelude as P
import qualified Data.List.Unrolled as U
import qualified Data.List.Unrolled3 as U3
import qualified Data.List.Unrolled4 as U4

import Criterion
import Criterion.Main
import Criterion.Config
import Criterion.MultiMap
import Data.Monoid

conf = defaultConfig {
--         cfgPlot = singleton KernelDensity (SVG 800 600)
--       , cfgPlotSameAxis = Last (Just True)
       }

main = defaultMainWith conf (return ()) [
        bgroup "sum-enum" [
         bench "standard"  
                  $ whnf (\n -> P.sum (P.enumFromTo (0+n::Int) (10000+n))) 10
        ,bench "unrolled1"
                  $ whnf (\n -> U.sum (U.enumFromToInt (0+n::Int) (10000+n))) 10
        ,bench "unrolled2"
                  $ whnf (\n -> U.sum (U.enumFromToInt2 (0+n::Int) (10000+n))) 10
        ,bench "unrolled generated 3"
                  $ whnf (\n -> U3.sum (U3.enumFromToL (0+n::Int) (10000+n))) 10
        ,bench "unrolled generated 4"
                  $ whnf (\n -> U4.sum (U4.enumFromToL (0+n::Int) (10000+n))) 10
        ]
       ,bgroup "sum-filter-enum" [
       -- These benchmarks are a bit flawed since the standard version
       -- benefits from fusion
         bench "standard"
                   $ whnf (\n -> P.sum (P.filter odd (P.enumFromTo (0+n::Int) (10000+n)))) 10
        ,bench "unrolled"
                   $ whnf (\n -> U.sum (U.filter odd (U.enumFromTo (0+n::Int) (10000+n)))) 10
        ,bench "unrolled reconstruct"
                   $ whnf (\n -> U.sum (U.filterR odd (U.enumFromTo (0+n::Int) (10000+n)))) 10
        ,bench "unrolled generated 3"
                   $ whnf (\n -> U3.sum (U3.filter odd (U3.enumFromToL (0+n::Int) (10000+n)))) 10
        ,bench "unrolled bad generated 3"
                   $ whnf (\n -> U3.sumBad (U3.filter odd (U3.enumFromToL (0+n::Int) (10000+n)))) 10
        ]
       ]
