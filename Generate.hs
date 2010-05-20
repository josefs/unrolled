{- The idea with this module is that it should generate modules with different
   amount of unrolling of lists. It should take a list of numbers and
   spit out modules which contains these amounts of unrolling

   I'd like to use haskell-src-exts to generate the code.
-}
module Generate where

import Language.Haskell.Exts.Syntax

genModule n = Module 
              noLoc -- No location, we're generating the code
              (ModuleName ("Data.List.Unrolled" ++ show n)) 
              [] -- No pragmas
              Nothing -- No warning text
              (Just []) -- Export list
              [] -- Import decls
              [] -- decls

noLoc = SrcLoc "Unknown" 0 0