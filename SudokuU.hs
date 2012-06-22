module SudokuU where

import Prelude hiding (zip,filter,length,repeat)
import Data.List.Unrolled
import Data.Array

type Index = ( Int,Int,Int,Int )

neighbours :: Index -> List Index
neighbours (a,b,c,d) = do
   i <- [ 0 .. 2 ] ; j <- [ 0 .. 2 ]
   [ (i,j,c,d), (a,b,i,j), (a,i,c,j) ]

type Matrix = Array Index (Either (List Int) Int)

solutions :: Matrix -> List Matrix
solutions m =
   case sort $ do
           ( i, Left xs ) <- assocs m
           return ( length xs, i, xs )
   of
       [] -> return m
       (_,i,xs) : _ -> do
           x <- xs
           solutions $ set (i,x) m

set :: (Index, Int) -> Matrix -> Matrix
set (i, x) m = accum ( \ e _ -> case e of
             Left ys -> Left $ filter ( /= x ) ys
             Right y -> Right y
           )
           ( m // [ (i, Right x ) ] )
           ( zip ( neighbours i ) $ repeat () )