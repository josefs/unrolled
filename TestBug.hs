{-# Language QuasiQuotes #-}
module TestBug where

import Language.Haskell.Exts
import Language.Haskell.Exts.QQ

-- Just using haskell-src-exts doesn't cause any problems

unit = TyTuple Boxed []

ty = [dec| quux :: (a,b) |]
