{-# LANGUAGE MagicHash,RankNTypes #-}
-----------------------------------------------------------------------------
-- |
-- Module      :  Data.List.Unrolled
-- Copyright   :  (c) The University of Glasgow 2001
--             :  (c) Josef Svenningsson 2009
-- License     :  BSD-style (see the file libraries/base/LICENSE)
-- 
-- Maintainer  :  libraries@haskell.org
-- Stability   :  stable
-- Portability :  portable
--
-- Operations on lists.
--
-----------------------------------------------------------------------------
module Data.List.Unrolled 

{-
    (
   -- * The unrolled list type
   List,

   -- * Basic functions
     (++)              -- :: [a] -> [a] -> [a]
   , head              -- :: [a] -> a
   , last              -- :: [a] -> a
   , tail              -- :: [a] -> [a]
   , init              -- :: [a] -> [a]
   , null              -- :: [a] -> Bool
   , length            -- :: [a] -> Int

   -- * List transformations
   , map               -- :: (a -> b) -> [a] -> [b]
   , reverse           -- :: [a] -> [a]

   , intersperse       -- :: a -> [a] -> [a]
   , intercalate       -- :: [a] -> [[a]] -> [a]
   , transpose         -- :: [[a]] -> [[a]]
   
   , subsequences      -- :: [a] -> [[a]]
   , permutations      -- :: [a] -> [[a]]

   -- * Reducing lists (folds)

   , foldl             -- :: (a -> b -> a) -> a -> [b] -> a
   , foldl'            -- :: (a -> b -> a) -> a -> [b] -> a
   , foldl1            -- :: (a -> a -> a) -> [a] -> a
   , foldl1'           -- :: (a -> a -> a) -> [a] -> a
   , foldr             -- :: (a -> b -> b) -> b -> [a] -> b
   , foldr1            -- :: (a -> a -> a) -> [a] -> a

   -- ** Special folds

   , concat            -- :: [[a]] -> [a]
   , concatMap         -- :: (a -> [b]) -> [a] -> [b]
   , and               -- :: [Bool] -> Bool
   , or                -- :: [Bool] -> Bool
   , any               -- :: (a -> Bool) -> [a] -> Bool
   , all               -- :: (a -> Bool) -> [a] -> Bool
   , sum               -- :: (Num a) => [a] -> a
   , product           -- :: (Num a) => [a] -> a
   , maximum           -- :: (Ord a) => [a] -> a
   , minimum           -- :: (Ord a) => [a] -> a

   -- * Building lists

   -- ** Scans
   , scanl             -- :: (a -> b -> a) -> a -> [b] -> [a]
   , scanl1            -- :: (a -> a -> a) -> [a] -> [a]
   , scanr             -- :: (a -> b -> b) -> b -> [a] -> [b]
   , scanr1            -- :: (a -> a -> a) -> [a] -> [a]

   -- ** Accumulating maps
   , mapAccumL         -- :: (a -> b -> (a,c)) -> a -> [b] -> (a,[c])
   , mapAccumR         -- :: (a -> b -> (a,c)) -> a -> [b] -> (a,[c])

   -- ** Infinite lists
   , iterate           -- :: (a -> a) -> a -> [a]
   , repeat            -- :: a -> [a]
   , replicate         -- :: Int -> a -> [a]
   , cycle             -- :: [a] -> [a]

   -- ** Unfolding
   , unfoldr           -- :: (b -> Maybe (a, b)) -> b -> [a]

   -- * Sublists

   -- ** Extracting sublists
   , take              -- :: Int -> [a] -> [a]
   , drop              -- :: Int -> [a] -> [a]
   , splitAt           -- :: Int -> [a] -> ([a], [a])

   , takeWhile         -- :: (a -> Bool) -> [a] -> [a]
   , dropWhile         -- :: (a -> Bool) -> [a] -> [a]
   , span              -- :: (a -> Bool) -> [a] -> ([a], [a])
   , break             -- :: (a -> Bool) -> [a] -> ([a], [a])

   , stripPrefix       -- :: Eq a => [a] -> [a] -> Maybe [a]

   , group             -- :: Eq a => [a] -> [[a]]

   , inits             -- :: [a] -> [[a]]
   , tails             -- :: [a] -> [[a]]

   -- ** Predicates
   , isPrefixOf        -- :: (Eq a) => [a] -> [a] -> Bool
   , isSuffixOf        -- :: (Eq a) => [a] -> [a] -> Bool
   , isInfixOf         -- :: (Eq a) => [a] -> [a] -> Bool

   -- * Searching lists

   -- ** Searching by equality
   , elem              -- :: a -> [a] -> Bool
   , notElem           -- :: a -> [a] -> Bool
   , lookup            -- :: (Eq a) => a -> [(a,b)] -> Maybe b

   -- ** Searching with a predicate
   , find              -- :: (a -> Bool) -> [a] -> Maybe a
   , filter            -- :: (a -> Bool) -> [a] -> [a]
   , partition         -- :: (a -> Bool) -> [a] -> ([a], [a])

   -- * Indexing lists
   -- | These functions treat a list @xs@ as a indexed collection,
   -- with indices ranging from 0 to @'length' xs - 1@.

   , (!!)              -- :: [a] -> Int -> a

   , elemIndex         -- :: (Eq a) => a -> [a] -> Maybe Int
   , elemIndices       -- :: (Eq a) => a -> [a] -> [Int]

   , findIndex         -- :: (a -> Bool) -> [a] -> Maybe Int
   , findIndices       -- :: (a -> Bool) -> [a] -> [Int]

   -- * Zipping and unzipping lists

   , zip               -- :: [a] -> [b] -> [(a,b)]
   , zip3
   , zip4, zip5, zip6, zip7

   , zipWith           -- :: (a -> b -> c) -> [a] -> [b] -> [c]
   , zipWith3
   , zipWith4, zipWith5, zipWith6, zipWith7

   , unzip             -- :: [(a,b)] -> ([a],[b])
   , unzip3
   , unzip4, unzip5, unzip6, unzip7

   -- * Special lists

   -- ** Functions on strings
   , lines             -- :: String   -> [String]
   , words             -- :: String   -> [String]
   , unlines           -- :: [String] -> String
   , unwords           -- :: [String] -> String

   -- ** \"Set\" operations

   , nub               -- :: (Eq a) => [a] -> [a]

   , delete            -- :: (Eq a) => a -> [a] -> [a]
   , (\\)              -- :: (Eq a) => [a] -> [a] -> [a]

   , union             -- :: (Eq a) => [a] -> [a] -> [a]
   , intersect         -- :: (Eq a) => [a] -> [a] -> [a]

   -- ** Ordered lists
   , sort              -- :: (Ord a) => [a] -> [a]
   , insert            -- :: (Ord a) => a -> [a] -> [a]

   -- * Generalized functions

   -- ** The \"@By@\" operations
   -- | By convention, overloaded functions have a non-overloaded
   -- counterpart whose name is suffixed with \`@By@\'.
   --
   -- It is often convenient to use these functions together with
   -- 'Data.Function.on', for instance @'sortBy' ('compare'
   -- \`on\` 'fst')@.

   -- *** User-supplied equality (replacing an @Eq@ context)
   -- | The predicate is assumed to define an equivalence.
   , nubBy             -- :: (a -> a -> Bool) -> [a] -> [a]
   , deleteBy          -- :: (a -> a -> Bool) -> a -> [a] -> [a]
   , deleteFirstsBy    -- :: (a -> a -> Bool) -> [a] -> [a] -> [a]
   , unionBy           -- :: (a -> a -> Bool) -> [a] -> [a] -> [a]
   , intersectBy       -- :: (a -> a -> Bool) -> [a] -> [a] -> [a]
   , groupBy           -- :: (a -> a -> Bool) -> [a] -> [[a]]

   -- *** User-supplied comparison (replacing an @Ord@ context)
   -- | The function is assumed to define a total ordering.
   , sortBy            -- :: (a -> a -> Ordering) -> [a] -> [a]
   , insertBy          -- :: (a -> a -> Ordering) -> a -> [a] -> [a]
   , maximumBy         -- :: (a -> a -> Ordering) -> [a] -> a
   , minimumBy         -- :: (a -> a -> Ordering) -> [a] -> a

   -- ** The \"@generic@\" operations
   -- | The prefix \`@generic@\' indicates an overloaded function that
   -- is a generalized version of a "Prelude" function.

   , genericLength     -- :: (Integral a) => [b] -> a
   , genericTake       -- :: (Integral a) => a -> [b] -> [b]
   , genericDrop       -- :: (Integral a) => a -> [b] -> [b]
   , genericSplitAt    -- :: (Integral a) => a -> [b] -> ([b], [b])
   , genericIndex      -- :: (Integral a) => [b] -> a -> b
   , genericReplicate  -- :: (Integral a) => a -> b -> [b]

    )
-}
    where

import Prelude (Eq(..),Ord(..),Functor(..),Monad(..),Enum(..)
               ,Num(..),Integral(..)
               ,Ordering(..)
               ,Bool(..),(&&),(||)
               ,Integer
               ,String(..),Char
               ,(.),error,otherwise,seq,flip,($)
               )
import qualified Prelude as P
import GHC.Base (Int(..),Int#,(+#),(-#),(<=#),(>=#),(<#),(>#),(==#),modInt#)

import Data.Maybe hiding (listToMaybe)
import Data.Char         ( isSpace )
{-
#ifdef __GLASGOW_HASKELL__
import GHC.Num
import GHC.Real
import GHC.List
import GHC.Base
#endif
-}
infix 5 \\ -- comment to fool cpp

data List a = Nil | Cons a (List a) | DCons a a (List a)

-- ----------------------------------------------------------------------------
-- List functions

-- | The 'stripPrefix' function drops the given prefix from a list.
-- It returns 'Nothing' if the list did not start with the prefix
-- given, or 'Just' the list after the prefix, if it does.
--
-- > stripPrefix "foo" "foobar" -> Just "bar"
-- > stripPrefix "foo" "foo" -> Just ""
-- > stripPrefix "foo" "barfoo" -> Nothing
-- > stripPrefix "foo" "barfoobaz" -> Nothing
stripPrefix :: Eq a => List a -> List a -> Maybe (List a)
stripPrefix Nil ys = Just ys
stripPrefix (Cons x xs) (Cons y ys)
    | x == y = stripPrefix xs ys
-- These two cases could do with more optimizations
stripPrefix (Cons x xs) (DCons y z ys)
    | x == y = stripPrefix xs (Cons z ys)
stripPrefix (DCons x z xs) (Cons y ys)
    | x == y = stripPrefix (Cons z xs) ys
stripPrefix (DCons x z xs) (DCons y q ys)
    | x == y && z == q = stripPrefix xs ys
stripPrefix _ _ = Nothing


-- | The 'elemIndex' function returns the index of the first element
-- in the given list which is equal (by '==') to the query element,
-- or 'Nothing' if there is no such element.
elemIndex       :: Eq a => a -> List a -> Maybe Int
elemIndex x     = findIndex (x==)

-- | The 'elemIndices' function extends 'elemIndex', by returning the
-- indices of all elements equal to the query element, in ascending order.
elemIndices     :: Eq a => a -> List a -> List Int
elemIndices x   = findIndices (x==)

-- | The 'find' function takes a predicate and a list and returns the
-- first element in the list matching the predicate, or 'Nothing' if
-- there is no such element.
find            :: (a -> Bool) -> List a -> Maybe a
find p          = listToMaybe . filter p

-- | The 'findIndex' function takes a predicate and a list and returns
-- the index of the first element in the list satisfying the predicate,
-- or 'Nothing' if there is no such element.
findIndex       :: (a -> Bool) -> List a -> Maybe Int
findIndex p     = listToMaybe . findIndices p

listToMaybe :: List a -> Maybe a
listToMaybe Nil           = Nothing
listToMaybe (Cons x _)    = Just x
listToMaybe (DCons x _ _) = Just x

-- | The 'findIndices' function extends 'findIndex', by returning the
-- indices of all elements satisfying the predicate, in ascending order.
findIndices      :: (a -> Bool) -> List a -> List Int
findIndices p ls = loop 0# ls
                 where
                   loop _ Nil = Nil
                   loop n (Cons x xs) 
                       | p x       = Cons (I# n) (loop (n +# 1#) xs)
                       | otherwise = loop (n +# 1#) xs
                   loop n (DCons x y xs) =
                       if p x then
                           if p y then
                               Cons (I# n) $
                               Cons (I# (n +# 1#)) $
                               loop (n +# 2#) xs
                           else
                               Cons (I# n) (loop (n +# 2#) xs)
                       else
                           if p y then
                               Cons (I# (n +# 1#)) (loop (n +# 2#) xs)
                           else
                               loop (n +# 2#) xs

-- | The 'isPrefixOf' function takes two lists and returns 'True'
-- iff the first list is a prefix of the second.
isPrefixOf              :: (Eq a) => List a -> List a -> Bool
isPrefixOf Nil _         =  True
isPrefixOf _   Nil       =  False
isPrefixOf (Cons x xs) (Cons y ys) = x == y && isPrefixOf xs ys
isPrefixOf (Cons x xs) (DCons y z ys) = x == y && isPrefixOf xs (Cons z ys)
isPrefixOf (DCons x z xs) (Cons y ys) = x == y && isPrefixOf (Cons z xs) ys
isPrefixOf (DCons x z xs) (DCons y q ys) = x == y && z == q && isPrefixOf xs ys

-- | The 'isSuffixOf' function takes two lists and returns 'True'
-- iff the first list is a suffix of the second.
-- Both lists must be finite.
isSuffixOf              :: (Eq a) => List a -> List a -> Bool
isSuffixOf x y          =  reverse x `isPrefixOf` reverse y

-- | The 'isInfixOf' function takes two lists and returns 'True'
-- iff the first list is contained, wholly and intact,
-- anywhere within the second.
--
-- Example:
--
-- >isInfixOf "Haskell" "I really like Haskell." -> True
-- >isInfixOf "Ial" "I really like Haskell." -> False
isInfixOf               :: (Eq a) => List a -> List a -> Bool
isInfixOf needle haystack = any (isPrefixOf needle) (tails haystack)

-- | The 'nub' function removes duplicate elements from a list.
-- In particular, it keeps only the first occurrence of each element.
-- (The name 'nub' means \`essence\'.)
-- It is a special case of 'nubBy', which allows the programmer to supply
-- their own equality test.
nub                     :: (Eq a) => List a -> List a
-- stolen from HBC
nub l                   = nub' l Nil
  where
    nub' Nil _          = Nil
    nub' (Cons x xs) ls
        | x `elem` ls   = nub' xs ls
        | otherwise     = Cons x (nub' xs (Cons x ls))
    nub' (DCons x y xs) ls =
        if x `elem` ls then
            if y `elem` ls then
                nub' xs ls
            else
                Cons y (nub' xs (Cons y ls))
        else
            if y `elem` ls then
                Cons x (nub' xs (Cons x ls))
            else
                DCons x y (nub' xs (DCons x y ls))

-- | The 'nubBy' function behaves just like 'nub', except it uses a
-- user-supplied equality predicate instead of the overloaded '=='
-- function.
nubBy                   :: (a -> a -> Bool) -> List a -> List a
nubBy eq l              = nubBy' l Nil
  where
    nubBy' Nil _        = Nil
    nubBy' (Cons y ys) xs
       | elem_by eq y xs = nubBy' ys xs
       | otherwise       = Cons y (nubBy' ys (Cons y xs))
    nubBy' (DCons x y ys) xs =
        if elem_by eq x xs then
           if elem_by eq y xs then
               nubBy' ys xs
           else
               Cons y (nubBy' ys (Cons y xs))
        else
            if elem_by eq y xs then
                Cons x (nubBy' ys (Cons x xs))
            else
                DCons x y (nubBy' ys (DCons x y xs))

-- Not exported:
-- Note that we keep the call to `eq` with arguments in the
-- same order as in the reference implementation
-- 'xs' is the list of things we've seen so far, 
-- 'y' is the potential new element
elem_by :: (a -> a -> Bool) -> a -> List a -> Bool
elem_by _  _ Nil         =  False
elem_by eq y (Cons x xs) =  y `eq` x || elem_by eq y xs
elem_by eq y (DCons x z xs) = y `eq` x || y `eq` z || elem_by eq y xs

-- | 'delete' @x@ removes the first occurrence of @x@ from its list argument.
-- For example,
--
-- > delete 'a' "banana" == "bnana"
--
-- It is a special case of 'deleteBy', which allows the programmer to
-- supply their own equality test.

delete                  :: (Eq a) => a -> List a -> List a
delete                  =  deleteBy (==)

-- | The 'deleteBy' function behaves like 'delete', but takes a
-- user-supplied equality predicate.
deleteBy                :: (a -> a -> Bool) -> a -> List a -> List a
deleteBy _  _ Nil         = Nil
deleteBy eq x (Cons y ys) = if x `eq` y then ys else Cons y (deleteBy eq x ys)
deleteBy eq x (DCons y z ys) = 
    if x `eq` y then
        Cons z ys
    else
        Cons y $
        if x `eq` z then
            ys
        else
            Cons z (deleteBy eq x ys)

-- | The '\\' function is list difference ((non-associative).
-- In the result of @xs@ '\\' @ys@, the first occurrence of each element of
-- @ys@ in turn (if any) has been removed from @xs@.  Thus
--
-- > (xs ++ ys) \\ xs == ys.
--
-- It is a special case of 'deleteFirstsBy', which allows the programmer
-- to supply their own equality test.

(\\)                    :: (Eq a) => List a -> List a -> List a
(\\)                    =  foldl (flip delete)

-- | The 'union' function returns the list union of the two lists.
-- For example,
--
-- > "dog" `union` "cow" == "dogcw"
--
-- Duplicates, and elements of the first list, are removed from the
-- the second list, but if the first list contains duplicates, so will
-- the result.
-- It is a special case of 'unionBy', which allows the programmer to supply
-- their own equality test.

union                   :: (Eq a) => List a -> List a -> List a
union                   = unionBy (==)

-- | The 'unionBy' function is the non-overloaded version of 'union'.
unionBy                 :: (a -> a -> Bool) -> List a -> List a -> List a
unionBy eq xs ys        =  xs ++ foldl (flip (deleteBy eq)) (nubBy eq ys) xs

-- | The 'intersect' function takes the list intersection of two lists.
-- For example,
--
-- > [1,2,3,4] `intersect` [2,4,6,8] == [2,4]
--
-- If the first list contains duplicates, so will the result.
--
-- > [1,2,2,3,4] `intersect` [6,4,4,2] == [2,2,4]
--
-- It is a special case of 'intersectBy', which allows the programmer to
-- supply their own equality test.

intersect               :: (Eq a) => List a -> List a -> List a
intersect               =  intersectBy (==)

-- | The 'intersectBy' function is the non-overloaded version of 'intersect'.
intersectBy             :: (a -> a -> Bool) -> List a -> List a -> List a
intersectBy eq xs ys    =  filter (\x -> any (eq x) ys) xs

-- | The 'intersperse' function takes an element and a list and
-- \`intersperses\' that element between the elements of the list.
-- For example,
--
-- > intersperse ',' "abcde" == "a,b,c,d,e"

intersperse             :: a -> List a -> List a
intersperse _   Nil             = Nil
intersperse _   (Cons x Nil)    = (Cons x Nil)
intersperse sep (Cons x xs)     = DCons x sep (intersperse sep xs)
intersperse sep (DCons x y Nil) = DCons x sep (Cons y Nil)
intersperse sep (DCons x y xs)  = DCons x sep (DCons y sep (intersperse sep xs))

-- | 'intercalate' @xs xss@ is equivalent to @('concat' ('intersperse' xs xss))@.
-- It inserts the list @xs@ in between the lists in @xss@ and concatenates the
-- result.
intercalate :: List a -> List (List a) -> List a
intercalate xs xss = concat (intersperse xs xss)

-- | The 'transpose' function transposes the rows and columns of its argument.
-- For example,
--
-- > transpose [[1,2,3],[4,5,6]] == [[1,4],[2,5],[3,6]]
{-
transpose               :: List (List a) -> List (List a)
transpose Nil             = Nil
transpose (Cons Nil xss) = transpose xss
transpose (Cons (Cons x xs) xss) = (x : [h | (h:_) <- xss]) : transpose (xs : [ t | (_:t) <- xss])
-}

-- | The 'partition' function takes a predicate a list and returns
-- the pair of lists of elements which do and do not satisfy the
-- predicate, respectively; i.e.,
--
-- > partition p xs == (filter p xs, filter (not . p) xs)

partition               :: (a -> Bool) -> List a -> (List a,List a)
{-# INLINE partition #-}
partition p xs = foldr (select p) (Nil,Nil) xs

select :: (a -> Bool) -> a -> (List a, List a) -> (List a, List a)
select p x ~(ts,fs) | p x       = (Cons x ts,fs)
                    | otherwise = (ts, Cons x fs)

-- | The 'mapAccumL' function behaves like a combination of 'map' and
-- 'foldl'; it applies a function to each element of a list, passing
-- an accumulating parameter from left to right, and returning a final
-- value of this accumulator together with the new list.
mapAccumL :: (acc -> x -> (acc, y)) -- Function of elt of input list
                                    -- and accumulator, returning new
                                    -- accumulator and elt of result list
          -> acc            -- Initial accumulator 
          -> List x         -- Input list
          -> (acc, List y)  -- Final accumulator and result list
mapAccumL _ s Nil         =  (s, Nil)
mapAccumL f s (Cons x xs) =  (s'',Cons y ys)
                           where (s', y ) = f s x
                                 (s'',ys) = mapAccumL f s' xs
-- Cheating
mapAccumL f s (DCons x y xs) = mapAccumL f s(Cons x (Cons y xs))

-- | The 'mapAccumR' function behaves like a combination of 'map' and
-- 'foldr'; it applies a function to each element of a list, passing
-- an accumulating parameter from right to left, and returning a final
-- value of this accumulator together with the new list.
mapAccumR :: (acc -> x -> (acc, y))     -- Function of elt of input list
                                        -- and accumulator, returning new
                                        -- accumulator and elt of result list
            -> acc              -- Initial accumulator
            -> List x           -- Input list
            -> (acc, List y)    -- Final accumulator and result list
mapAccumR _ s Nil         =  (s, Nil)
mapAccumR f s (Cons x xs) =  (s'', Cons y ys)
                           where (s'',y ) = f s' x
                                 (s', ys) = mapAccumR f s xs
mapAccumR f s (DCons x y xs) = mapAccumR f s (Cons x (Cons y xs))

-- | The 'insert' function takes an element and a list and inserts the
-- element into the list at the last position where it is still less
-- than or equal to the next element.  In particular, if the list
-- is sorted before the call, the result will also be sorted.
-- It is a special case of 'insertBy', which allows the programmer to
-- supply their own comparison function.
insert :: Ord a => a -> List a -> List a
insert e ls = insertBy (compare) e ls

-- | The non-overloaded version of 'insert'.
insertBy :: (a -> a -> Ordering) -> a -> List a -> List a
insertBy _   x Nil = Cons x Nil
insertBy cmp x ys@(Cons y ys')
 = case cmp x y of
     GT -> Cons y (insertBy cmp x ys')
     _  -> Cons x ys
insertBy cmp x ys@(DCons y z ys')
    = case cmp x y of
        GT -> case cmp x z of
                GT -> DCons y z (insertBy cmp x ys')
                _  -> DCons y x (Cons z ys')
        _  -> Cons x ys

-- | 'maximum' returns the maximum value from a list,
-- which must be non-empty, finite, and of an ordered type.
-- It is a special case of 'Data.List.maximumBy', which allows the
-- programmer to supply their own comparison function.
maximum                 :: (Ord a) => List a -> a
maximum Nil             =  errorEmptyList "maximum"
maximum xs              =  foldl1 max xs

{-# RULES
  "maximumInt"     maximum = (strictMaximum :: List Int     -> Int);
  "maximumInteger" maximum = (strictMaximum :: List Integer -> Integer)
 #-}

-- We can't make the overloaded version of maximum strict without
-- changing its semantics (max might not be strict), but we can for
-- the version specialised to 'Int'.
strictMaximum           :: (Ord a) => List a -> a
strictMaximum Nil       =  errorEmptyList "maximum"
strictMaximum xs        =  foldl1' max xs

-- | 'minimum' returns the minimum value from a list,
-- which must be non-empty, finite, and of an ordered type.
-- It is a special case of 'Data.List.minimumBy', which allows the
-- programmer to supply their own comparison function.
minimum                 :: (Ord a) => List a -> a
minimum Nil             =  errorEmptyList "minimum"
minimum xs              =  foldl1 min xs

{-# RULES
  "minimumInt"     minimum = (strictMinimum :: List Int     -> Int);
  "minimumInteger" minimum = (strictMinimum :: List Integer -> Integer)
 #-}

strictMinimum           :: (Ord a) => List a -> a
strictMinimum Nil       =  errorEmptyList "minimum"
strictMinimum xs        =  foldl1' min xs

-- | The 'maximumBy' function takes a comparison function and a list
-- and returns the greatest element of the list by the comparison function.
-- The list must be finite and non-empty.
maximumBy               :: (a -> a -> Ordering) -> List a -> a
maximumBy _ Nil         =  error "List.maximumBy: empty list"
maximumBy cmp xs        =  foldl1 maxBy xs
                        where
                           maxBy x y = case cmp x y of
                                       GT -> x
                                       _  -> y

-- | The 'minimumBy' function takes a comparison function and a list
-- and returns the least element of the list by the comparison function.
-- The list must be finite and non-empty.
minimumBy               :: (a -> a -> Ordering) -> List a -> a
minimumBy _ Nil         =  error "List.minimumBy: empty list"
minimumBy cmp xs        =  foldl1 minBy xs
                        where
                           minBy x y = case cmp x y of
                                       GT -> y
                                       _  -> x

-- | The 'genericLength' function is an overloaded version of 'length'.  In
-- particular, instead of returning an 'Int', it returns any type which is
-- an instance of 'Num'.  It is, however, less efficient than 'length'.
genericLength           :: (Num i) => List b -> i
genericLength Nil           =  0
genericLength (Cons _ l)    =  1 + genericLength l
genericLength (DCons _ _ l) =  2 + genericLength l

{-# RULES
  "genericLengthInt"     genericLength = (strictGenericLength :: List a -> Int);
  "genericLengthInteger" genericLength = (strictGenericLength :: List a -> Integer);
 #-}

strictGenericLength     :: (Num i) => List b -> i
strictGenericLength l   =  gl l 0
    where
      gl Nil a            = a
      gl (Cons _ xs) a    = let a' = a + 1 in a' `seq` gl xs a'
      gl (DCons _ _ xs) a = let a' = a + 2 in a' `seq` gl xs a'

-- | The 'genericTake' function is an overloaded version of 'take', which
-- accepts any 'Integral' value as the number of elements to take.
genericTake             :: (Integral i) => i -> List a -> List a
genericTake n _ | n <= 0  = Nil
genericTake _ Nil         = Nil
genericTake n (Cons x xs) = Cons x (genericTake (n-1) xs)
genericTake n (DCons x y xs) = genericTake n (Cons x (Cons y xs)) -- Sloppy

-- | The 'genericDrop' function is an overloaded version of 'drop', which
-- accepts any 'Integral' value as the number of elements to drop.
genericDrop             :: (Integral i) => i -> List a -> List a
genericDrop n xs | n <= 0 = xs
genericDrop _ Nil         = Nil
genericDrop n (Cons _ xs) = genericDrop (n-1) xs
genericDrop n (DCons x y xs) = genericDrop n (Cons x (Cons y xs)) -- Sloppy


-- | The 'genericSplitAt' function is an overloaded version of 'splitAt', which
-- accepts any 'Integral' value as the position at which to split.
genericSplitAt          :: (Integral i) => i -> List b -> (List b,List b)
genericSplitAt n xs | n <= 0 =  (Nil,xs)
genericSplitAt _ Nil         =  (Nil,Nil)
genericSplitAt n (Cons x xs) =  (Cons x xs',xs'') where
    (xs',xs'') = genericSplitAt (n-1) xs
genericSplitAt n (DCons x y xs) = genericSplitAt n (Cons x (Cons y xs)) -- Sloppy

-- | The 'genericIndex' function is an overloaded version of '!!', which
-- accepts any 'Integral' value as the index.
genericIndex :: (Integral a) => List b -> a -> b
genericIndex (Cons x _)  0 = x
genericIndex (Cons _ xs) n
 | n > 0     = genericIndex xs (n-1)
 | otherwise = error "List.genericIndex: negative argument."
genericIndex (DCons x _ _)  0 = x
genericIndex (DCons _ x _)  1 = x
genericIndex (DCons _ _ xs) n 
 | n > 0     = genericIndex xs (n-2)
 | otherwise = error "List.genericIndex: negative argument."
genericIndex _ _      = error "List.genericIndex: index too large."

-- | The 'genericReplicate' function is an overloaded version of 'replicate',
-- which accepts any 'Integral' value as the number of repetitions to make.
genericReplicate        :: (Integral i) => i -> a -> List a
genericReplicate n x    =  genericTake n (repeat x)

-- | The 'zip4' function takes four lists and returns a list of
-- quadruples, analogous to 'zip'.
zip4                    :: List a -> List b -> List c -> List d -> List (a,b,c,d)
zip4                    =  zipWith4 (,,,)
{-
-- | The 'zip5' function takes five lists and returns a list of
-- five-tuples, analogous to 'zip'.
zip5                    :: [a] -> [b] -> [c] -> [d] -> [e] -> [(a,b,c,d,e)]
zip5                    =  zipWith5 (,,,,)

-- | The 'zip6' function takes six lists and returns a list of six-tuples,
-- analogous to 'zip'.
zip6                    :: [a] -> [b] -> [c] -> [d] -> [e] -> [f] ->
                              [(a,b,c,d,e,f)]
zip6                    =  zipWith6 (,,,,,)

-- | The 'zip7' function takes seven lists and returns a list of
-- seven-tuples, analogous to 'zip'.
zip7                    :: [a] -> [b] -> [c] -> [d] -> [e] -> [f] ->
                              [g] -> [(a,b,c,d,e,f,g)]
zip7                    =  zipWith7 (,,,,,,)
-}
-- | The 'zipWith4' function takes a function which combines four
-- elements, as well as four lists and returns a list of their point-wise
-- combination, analogous to 'zipWith'.
zipWith4                :: (a->b->c->d->e) -> List a->List b->List c->List d->List e
zipWith4 z (Cons a as) (Cons b bs) (Cons c cs) (Cons d ds)
                        =  Cons (z a b c d) (zipWith4 z as bs cs ds)
zipWith4 _ _ _ _ _      =  Nil
{-
-- | The 'zipWith5' function takes a function which combines five
-- elements, as well as five lists and returns a list of their point-wise
-- combination, analogous to 'zipWith'.
zipWith5                :: (a->b->c->d->e->f) ->
                           [a]->[b]->[c]->[d]->[e]->[f]
zipWith5 z (a:as) (b:bs) (c:cs) (d:ds) (e:es)
                        =  z a b c d e : zipWith5 z as bs cs ds es
zipWith5 _ _ _ _ _ _    = []

-- | The 'zipWith6' function takes a function which combines six
-- elements, as well as six lists and returns a list of their point-wise
-- combination, analogous to 'zipWith'.
zipWith6                :: (a->b->c->d->e->f->g) ->
                           [a]->[b]->[c]->[d]->[e]->[f]->[g]
zipWith6 z (a:as) (b:bs) (c:cs) (d:ds) (e:es) (f:fs)
                        =  z a b c d e f : zipWith6 z as bs cs ds es fs
zipWith6 _ _ _ _ _ _ _  = []

-- | The 'zipWith7' function takes a function which combines seven
-- elements, as well as seven lists and returns a list of their point-wise
-- combination, analogous to 'zipWith'.
zipWith7                :: (a->b->c->d->e->f->g->h) ->
                           [a]->[b]->[c]->[d]->[e]->[f]->[g]->[h]
zipWith7 z (a:as) (b:bs) (c:cs) (d:ds) (e:es) (f:fs) (g:gs)
                   =  z a b c d e f g : zipWith7 z as bs cs ds es fs gs
zipWith7 _ _ _ _ _ _ _ _ = []
-}
{-
-- | The 'unzip4' function takes a list of quadruples and returns four
-- lists, analogous to 'unzip'.
unzip4                  :: [(a,b,c,d)] -> ([a],[b],[c],[d])
unzip4                  =  foldr (\(a,b,c,d) ~(as,bs,cs,ds) ->
                                        (a:as,b:bs,c:cs,d:ds))
                                 ([],[],[],[])

-- | The 'unzip5' function takes a list of five-tuples and returns five
-- lists, analogous to 'unzip'.
unzip5                  :: [(a,b,c,d,e)] -> ([a],[b],[c],[d],[e])
unzip5                  =  foldr (\(a,b,c,d,e) ~(as,bs,cs,ds,es) ->
                                        (a:as,b:bs,c:cs,d:ds,e:es))
                                 ([],[],[],[],[])

-- | The 'unzip6' function takes a list of six-tuples and returns six
-- lists, analogous to 'unzip'.
unzip6                  :: [(a,b,c,d,e,f)] -> ([a],[b],[c],[d],[e],[f])
unzip6                  =  foldr (\(a,b,c,d,e,f) ~(as,bs,cs,ds,es,fs) ->
                                        (a:as,b:bs,c:cs,d:ds,e:es,f:fs))
                                 ([],[],[],[],[],[])

-- | The 'unzip7' function takes a list of seven-tuples and returns
-- seven lists, analogous to 'unzip'.
unzip7          :: [(a,b,c,d,e,f,g)] -> ([a],[b],[c],[d],[e],[f],[g])
unzip7          =  foldr (\(a,b,c,d,e,f,g) ~(as,bs,cs,ds,es,fs,gs) ->
                                (a:as,b:bs,c:cs,d:ds,e:es,f:fs,g:gs))
                         ([],[],[],[],[],[],[])

-}
-- | The 'deleteFirstsBy' function takes a predicate and two lists and
-- returns the first list with the first occurrence of each element of
-- the second list removed.
deleteFirstsBy          :: (a -> a -> Bool) -> List a -> List a -> List a
deleteFirstsBy eq       =  foldl (flip (deleteBy eq))

-- | The 'group' function takes a list and returns a list of lists such
-- that the concatenation of the result is equal to the argument.  Moreover,
-- each sublist in the result contains only equal elements.  For example,
--
-- > group "Mississippi" = ["M","i","ss","i","ss","i","pp","i"]
--
-- It is a special case of 'groupBy', which allows the programmer to supply
-- their own equality test.
group                   :: Eq a => List a -> List (List a)
group                   =  groupBy (==)

-- | The 'groupBy' function is the non-overloaded version of 'group'.
groupBy                 :: (a -> a -> Bool) -> List a -> List (List a)
groupBy _  Nil            =  Nil
groupBy eq (Cons x xs)    =  Cons (Cons x ys) (groupBy eq zs)
    where (ys,zs) = span (eq x) xs
groupBy eq (DCons x y xs) = Cons (Cons x ys) (groupBy eq zs)
    where (ys,zs) = span (eq x) (Cons y xs)

-- | The 'inits' function returns all initial segments of the argument,
-- shortest first.  For example,
--
-- > inits "abc" == ["","a","ab","abc"]
--
inits                   :: List a -> List (List a)
inits Nil               = Cons Nil Nil
inits (Cons x xs)       = Cons Nil (map (Cons x) (inits xs))
inits (DCons x y xs)    = Cons Nil (Cons (Cons x Nil) (map (Cons x . Cons y) (inits xs)))

-- | The 'tails' function returns all final segments of the argument,
-- longest first.  For example,
--
-- > tails "abc" == ["abc", "bc", "c",""]
--
tails                    :: List a -> List (List a)
tails Nil                = Cons Nil Nil
tails xxs@(Cons _ xs)    = Cons xxs (tails xs)
tails xxs@(DCons _ x xs) = DCons xxs (Cons x xs) (tails xs)

{-
-- | The 'subsequences' function returns the list of all subsequences of the argument.
--
-- > subsequences "abc" == ["","a","b","ab","c","ac","bc","abc"]
subsequences            :: List a -> List (List a)
subsequences xs         =  Cons Nil (nonEmptySubsequences xs)

-- | The 'nonEmptySubsequences' function returns the list of all subsequences of the argument,
--   except for the empty list.
--
-- > nonEmptySubsequences "abc" == ["a","b","ab","c","ac","bc","abc"]
nonEmptySubsequences         :: [a] -> [[a]]
nonEmptySubsequences []      =  []
nonEmptySubsequences (x:xs)  =  [x] : foldr f [] (nonEmptySubsequences xs)
  where f ys r = ys : (x : ys) : r


-- | The 'permutations' function returns the list of all permutations of the argument.
--
-- > permutations "abc" == ["abc","bac","cba","bca","cab","acb"]
permutations            :: [a] -> [[a]]
permutations xs0        =  xs0 : perms xs0 []
  where
    perms []     _  = []
    perms (t:ts) is = foldr interleave (perms ts (t:is)) (permutations is)
      where interleave    xs     r = let (_,zs) = interleave' id xs r in zs
            interleave' _ []     r = (ts, r)
            interleave' f (y:ys) r = let (us,zs) = interleave' (f . (y:)) ys r
                                     in  (y:us, f (t:y:us) : zs)


------------------------------------------------------------------------------
-- Quick Sort algorithm taken from HBC's QSort library.

-- | The 'sort' function implements a stable sorting algorithm.
-- It is a special case of 'sortBy', which allows the programmer to supply
-- their own comparison function.
sort :: (Ord a) => [a] -> [a]

-- | The 'sortBy' function is the non-overloaded version of 'sort'.
sortBy :: (a -> a -> Ordering) -> [a] -> [a]

#ifdef USE_REPORT_PRELUDE
sort = sortBy compare
sortBy cmp = foldr (insertBy cmp) []
#else

sortBy cmp l = mergesort cmp l
sort l = mergesort compare l

{-
Quicksort replaced by mergesort, 14/5/2002.

From: Ian Lynagh <igloo@earth.li>

I am curious as to why the List.sort implementation in GHC is a
quicksort algorithm rather than an algorithm that guarantees n log n
time in the worst case? I have attached a mergesort implementation along
with a few scripts to time it's performance, the results of which are
shown below (* means it didn't finish successfully - in all cases this
was due to a stack overflow).

If I heap profile the random_list case with only 10000 then I see
random_list peaks at using about 2.5M of memory, whereas in the same
program using List.sort it uses only 100k.

Input style     Input length     Sort data     Sort alg    User time
stdin           10000            random_list   sort        2.82
stdin           10000            random_list   mergesort   2.96
stdin           10000            sorted        sort        31.37
stdin           10000            sorted        mergesort   1.90
stdin           10000            revsorted     sort        31.21
stdin           10000            revsorted     mergesort   1.88
stdin           100000           random_list   sort        *
stdin           100000           random_list   mergesort   *
stdin           100000           sorted        sort        *
stdin           100000           sorted        mergesort   *
stdin           100000           revsorted     sort        *
stdin           100000           revsorted     mergesort   *
func            10000            random_list   sort        0.31
func            10000            random_list   mergesort   0.91
func            10000            sorted        sort        19.09
func            10000            sorted        mergesort   0.15
func            10000            revsorted     sort        19.17
func            10000            revsorted     mergesort   0.16
func            100000           random_list   sort        3.85
func            100000           random_list   mergesort   *
func            100000           sorted        sort        5831.47
func            100000           sorted        mergesort   2.23
func            100000           revsorted     sort        5872.34
func            100000           revsorted     mergesort   2.24
-}

mergesort :: (a -> a -> Ordering) -> [a] -> [a]
mergesort cmp = mergesort' cmp . map wrap

mergesort' :: (a -> a -> Ordering) -> [[a]] -> [a]
mergesort' _   [] = []
mergesort' _   [xs] = xs
mergesort' cmp xss = mergesort' cmp (merge_pairs cmp xss)

merge_pairs :: (a -> a -> Ordering) -> [[a]] -> [[a]]
merge_pairs _   [] = []
merge_pairs _   [xs] = [xs]
merge_pairs cmp (xs:ys:xss) = merge cmp xs ys : merge_pairs cmp xss

merge :: (a -> a -> Ordering) -> [a] -> [a] -> [a]
merge _   [] ys = ys
merge _   xs [] = xs
merge cmp (x:xs) (y:ys)
 = case x `cmp` y of
        GT -> y : merge cmp (x:xs)   ys
        _  -> x : merge cmp    xs (y:ys)

wrap :: a -> [a]
wrap x = [x]

{-
OLD: qsort version

-- qsort is stable and does not concatenate.
qsort :: (a -> a -> Ordering) -> [a] -> [a] -> [a]
qsort _   []     r = r
qsort _   [x]    r = x:r
qsort cmp (x:xs) r = qpart cmp x xs [] [] r

-- qpart partitions and sorts the sublists
qpart :: (a -> a -> Ordering) -> a -> [a] -> [a] -> [a] -> [a] -> [a]
qpart cmp x [] rlt rge r =
    -- rlt and rge are in reverse order and must be sorted with an
    -- anti-stable sorting
    rqsort cmp rlt (x:rqsort cmp rge r)
qpart cmp x (y:ys) rlt rge r =
    case cmp x y of
        GT -> qpart cmp x ys (y:rlt) rge r
        _  -> qpart cmp x ys rlt (y:rge) r

-- rqsort is as qsort but anti-stable, i.e. reverses equal elements
rqsort :: (a -> a -> Ordering) -> [a] -> [a] -> [a]
rqsort _   []     r = r
rqsort _   [x]    r = x:r
rqsort cmp (x:xs) r = rqpart cmp x xs [] [] r

rqpart :: (a -> a -> Ordering) -> a -> [a] -> [a] -> [a] -> [a] -> [a]
rqpart cmp x [] rle rgt r =
    qsort cmp rle (x:qsort cmp rgt r)
rqpart cmp x (y:ys) rle rgt r =
    case cmp y x of
        GT -> rqpart cmp x ys rle (y:rgt) r
        _  -> rqpart cmp x ys (y:rle) rgt r
-}

#endif /* USE_REPORT_PRELUDE */
-}
-- | The 'unfoldr' function is a \`dual\' to 'foldr': while 'foldr'
-- reduces a list to a summary value, 'unfoldr' builds a list from
-- a seed value.  The function takes the element and returns 'Nothing'
-- if it is done producing the list or returns 'Just' @(a,b)@, in which
-- case, @a@ is a prepended to the list and @b@ is used as the next
-- element in a recursive call.  For example,
--
-- > iterate f == unfoldr (\x -> Just (x, f x))
--
-- In some cases, 'unfoldr' can undo a 'foldr' operation:
--
-- > unfoldr f' (foldr f z xs) == xs
--
-- if the following holds:
--
-- > f' (f x y) = Just (x,y)
-- > f' z       = Nothing
--
-- A simple use of unfoldr:
--
-- > unfoldr (\b -> if b == 0 then Nothing else Just (b, b-1)) 10
-- >  [10,9,8,7,6,5,4,3,2,1]
--
unfoldr      :: (b -> Maybe (a, b)) -> b -> List a
unfoldr f b  =
  case f b of
   Just (a,new_b) -> Cons a (unfoldr f new_b)
   Nothing        -> Nil

-- -----------------------------------------------------------------------------

-- | A strict version of 'foldl'.
foldl'           :: (a -> b -> a) -> a -> List b -> a
foldl' f z0 xs0 = lgo z0 xs0
    where lgo z Nil            = z
          lgo z (Cons  x   xs) = let z' = f z x 
                                 in z' `seq` lgo z' xs
          lgo z (DCons x y xs) = let z'  = f z x 
                                     z'' = f z' y
                                 in z'' `seq` lgo z'' xs

-- | 'foldl1' is a variant of 'foldl' that has no starting value argument,
-- and thus must be applied to non-empty lists.
foldl1                  :: (a -> a -> a) -> List a -> a
foldl1 f (Cons x xs)    =  foldl f x xs
foldl1 _ Nil            =  errorEmptyList "foldl1"

-- | A strict version of 'foldl1'
foldl1'                  :: (a -> a -> a) -> List a -> a
foldl1' f (Cons x xs)    =  foldl' f x xs
foldl1' _ Nil            =  errorEmptyList "foldl1'"

-- -----------------------------------------------------------------------------
-- List sum and product

{-# SPECIALISE sum     :: List Int -> Int #-}
{-# SPECIALISE sum     :: List Integer -> Integer #-}
{-# SPECIALISE product :: List Int -> Int #-}
{-# SPECIALISE product :: List Integer -> Integer #-}
-- | The 'sum' function computes the sum of a finite list of numbers.
sum                     :: (Num a) => List a -> a
-- | The 'product' function computes the product of a finite list of numbers.
product                 :: (Num a) => List a -> a

sum     l       = sum' l 0
  where
    sum' Nil            a = a
    sum' (Cons x xs)    a = sum' xs (a+x)
    sum' (DCons x y xs) a = sum' xs (a+x+y)
product l       = prod l 1
  where
    prod Nil            a = a
    prod (Cons x xs)    a = prod xs (a*x)
    prod (DCons x y xs) a = prod xs (a*x*y)
{-
-- -----------------------------------------------------------------------------
-- Functions on strings

-- | 'lines' breaks a string up into a list of strings at newline
-- characters.  The resulting strings do not contain newlines.
lines                   :: String -> [String]
lines ""                =  []
lines s                 =  let (l, s') = break (== '\n') s
                           in  l : case s' of
                                        []      -> []
                                        (_:s'') -> lines s''

-- | 'unlines' is an inverse operation to 'lines'.
-- It joins lines, after appending a terminating newline to each.
unlines                 :: [String] -> String
#ifdef USE_REPORT_PRELUDE
unlines                 =  concatMap (++ "\n")
#else
-- HBC version (stolen)
-- here's a more efficient version
unlines [] = []
unlines (l:ls) = l ++ '\n' : unlines ls
#endif

-- | 'words' breaks a string up into a list of words, which were delimited
-- by white space.
words                   :: String -> [String]
words s                 =  case dropWhile {-partain:Char.-}isSpace s of
                                "" -> []
                                s' -> w : words s''
                                      where (w, s'') =
                                             break {-partain:Char.-}isSpace s'

-- | 'unwords' is an inverse operation to 'words'.
-- It joins words with separating spaces.
unwords                 :: [String] -> String
#ifdef USE_REPORT_PRELUDE
unwords []              =  ""
unwords ws              =  foldr1 (\w s -> w ++ ' ':s) ws
#else
-- HBC version (stolen)
-- here's a more efficient version
unwords []              =  ""
unwords [w]             = w
unwords (w:ws)          = w ++ ' ' : unwords ws
#endif

errorEmptyList :: String -> a
errorEmptyList fun =
  error ("Prelude." ++ fun ++ ": empty list")


-}

{- Here begins GHC.List -}

infixl 9  !!
infix  4 `elem`, `notElem`
{-

%*********************************************************
%*                                                      *
\subsection{List-manipulation functions}
%*                                                      *
%*********************************************************

\begin{code}
-}
-- | Extract the first element of a list, which must be non-empty.
head                    :: List a -> a
head (Cons x _)         =  x
head (DCons x _ _)      = x
head Nil                =  badHead

badHead :: a
badHead = errorEmptyList "head"

-- This rule is useful in cases like 
--      head [y | (x,y) <- ps, x==t]
{- RULES
"head/build"    forall (g::forall b.(a->b->b)->b->b) .
                head (build g) = g (\x _ -> x) badHead
"head/augment"  forall xs (g::forall b. (a->b->b) -> b -> b) . 
                head (augment g xs) = g (\x _ -> x) (head xs)
 -}

-- | Extract the elements after the head of a list, which must be non-empty.
tail                    :: List a -> List a
tail (Cons _ xs)        = xs
tail (DCons _ x xs)     = Cons x xs
tail Nil                = errorEmptyList "tail"

-- | Extract the last element of a list, which must be finite and non-empty.
last                    :: List a -> a
last Nil                = errorEmptyList "last"
last (Cons x xs)        = last' x xs
last (DCons _ x xs)     = last' x xs

-- eliminate repeated cases
last' y Nil            = y
last' _ (Cons    y ys) = last' y ys
last' _ (DCons _ y ys) = last' y ys


-- | Return all the elements of a list except the last one.
-- The list must be finite and non-empty.
init                    :: [a] -> [a]
#ifdef USE_REPORT_PRELUDE
init [x]                =  []
init (x:xs)             =  x : init xs
init []                 =  errorEmptyList "init"
#else
-- eliminate repeated cases
init []                 =  errorEmptyList "init"
init (x:xs)             =  init' x xs
  where init' _ []     = []
        init' y (z:zs) = y : init' z zs
#endif

-- | Test whether a list is empty.
null                    :: List a -> Bool
null Nil                =  True
null (Cons _ _)         =  False
null (DCons _ _ _)      =  False

-- | 'length' returns the length of a finite list as an 'Int'.
-- It is an instance of the more general 'Data.List.genericLength',
-- the result type of which may be any kind of number.
length                  :: List a -> Int
length l                =  len l 0#
  where
    len :: List a -> Int# -> Int
    len Nil            a# = I# a#
    len (Cons _ xs)    a# = len xs (a# +# 1#)
    len (DCons _ _ xs) a# = len xs (a# +# 2#)

-- | 'filter', applied to a predicate and a list, returns the list of
-- those elements that satisfy the predicate; i.e.,
--
-- > filter p xs = [ x | x <- xs, p x]
{-
filter :: (a -> Bool) -> [a] -> [a]
filter _pred []    = []
filter pred (x:xs)
  | pred x         = x : filter pred xs
  | otherwise      = filter pred xs
-}

-- There are two versions here, one for realigning and one that just doesn't
-- care. The don't care version is the default.
-- In my benchmark 'sum (filter odd (fromEnumTo ...))' they make no difference
-- what so ever.

filter p Nil = Nil
filter p (Cons a fs) | p a = Cons a (filter p fs)
                     | otherwise = filter p fs
filter p (DCons a b fs) =
    if p a then
        if p b then
            DCons a b (filter p fs)
        else
            Cons a (filter p fs)
    else
        if p b then
            Cons b (filter p fs)
        else
            filter p fs

filterR _p Nil = Nil
filterR p  (Cons a fs) | p a       = filter1 a p fs
                       | otherwise = filterR p fs
filterR p (DCons a b fs) =
    if p a then
        if p b then
            DCons a b (filterR p fs)
        else
            filter1 a p fs
    else
        if p b then
            filter1 b p fs
        else
            filterR p fs

filter1 a p Nil = Cons a Nil
filter1 a p (Cons b fs) | p b       = DCons a b (filterR p fs)
                        | otherwise = filter1 a p fs
filter1 a p (DCons b c fs) =
    if p b then
        if p c then
            DCons a b (filter1 c p fs)
        else
            DCons a b (filterR p fs)
    else
        if p c then
            DCons a c (filterR p fs)
        else 
            filter1 a p fs


{-# NOINLINE [0] filterFB #-}
filterFB :: (a -> b -> b) -> (a -> Bool) -> a -> b -> b
filterFB c p x r | p x       = x `c` r
                 | otherwise = r

{- RULES
"filter"     [~1] forall p xs.  filter p xs = build (\c n -> foldr (filterFB c p) n xs)
"filterList" [1]  forall p.     foldr (filterFB (:) p) [] = filter p
"filterFB"        forall c p q. filterFB (filterFB c p) q = filterFB c (\x -> q x && p x)
 -}

-- Note the filterFB rule, which has p and q the "wrong way round" in the RHS.
--     filterFB (filterFB c p) q a b
--   = if q a then filterFB c p a b else b
--   = if q a then (if p a then c a b else b) else b
--   = if q a && p a then c a b else b
--   = filterFB c (\x -> q x && p x) a b
-- I originally wrote (\x -> p x && q x), which is wrong, and actually
-- gave rise to a live bug report.  SLPJ.


-- | 'foldl', applied to a binary operator, a starting value (typically
-- the left-identity of the operator), and a list, reduces the list
-- using the binary operator, from left to right:
--
-- > foldl f z [x1, x2, ..., xn] == (...((z `f` x1) `f` x2) `f`...) `f` xn
--
-- The list must be finite.

-- We write foldl as a non-recursive thing, so that it
-- can be inlined, and then (often) strictness-analysed,
-- and hence the classic space leak on foldl (+) 0 xs

foldl        :: (a -> b -> a) -> a -> List b -> a
foldl f z0 xs0 = lgo z0 xs0
             where
                lgo z Nil            =  z
                lgo z (Cons x    xs) = lgo (f z x) xs
                lgo z (DCons x y xs) = lgo (f (f z x) y) xs

-- | 'scanl' is similar to 'foldl', but returns a list of successive
-- reduced values from the left:
--
-- > scanl f z [x1, x2, ...] == [z, z `f` x1, (z `f` x1) `f` x2, ...]
--
-- Note that
--
-- > last (scanl f z xs) == foldl f z xs.

-- This is problematic because scanl is lazy in the input list. This
-- means that we cannot inspect the list first. But maybe I can 
-- do some trick where I use a helper function once I know what the 
-- list looks like.

scanl :: (a -> b -> a) -> a -> List b -> List a
scanl f q ls =  Cons q (case ls of
                          Nil          -> Nil
                          Cons x xs    -> scanl f (f q x) xs
                          DCons x y xs -> 
                              let fqx = f q x in
                              Cons fqx (scanl f (f fqx y) xs))


-- | 'scanl1' is a variant of 'scanl' that has no starting value argument:
--
-- > scanl1 f [x1, x2, ...] == [x1, x1 `f` x2, ...]

scanl1                  :: (a -> a -> a) -> List a -> List a
scanl1 f (DCons x y xs) =  Cons x (scanl f (f x y) xs)
scanl1 f (Cons  x   xs) =  scanl f x xs
scanl1 _ Nil            =  Nil

-- foldr, foldr1, scanr, and scanr1 are the right-to-left duals of the
-- above functions.

-- | 'foldr1' is a variant of 'foldr' that has no starting value argument,
-- and thus must be applied to non-empty lists.

foldr1                  :: (a -> a -> a) -> List a -> a
foldr1 _ (Cons  x  Nil) =  x
foldr1 f (Cons  x   xs) =  f x (foldr1 f xs)
foldr1 f (DCons x y xs) =  f x (f y (foldr1 f xs))
foldr1 _ Nil            =  errorEmptyList "foldr1"

-- | 'scanr' is the right-to-left dual of 'scanl'.
-- Note that
--
-- > head (scanr f z xs) == foldr f z xs.

scanr                   :: (a -> b -> b) -> b -> [a] -> [b]
scanr _ q0 []           =  [q0]
scanr f q0 (x:xs)       =  f x q : qs
                           where qs@(q:_) = scanr f q0 xs 

-- | 'scanr1' is a variant of 'scanr' that has no starting value argument.

scanr1                  :: (a -> a -> a) -> [a] -> [a]
scanr1 _ []             =  []
scanr1 _ [x]            =  [x]
scanr1 f (x:xs)         =  f x q : qs
                           where qs@(q:_) = scanr1 f xs 

-- | 'iterate' @f x@ returns an infinite list of repeated applications
-- of @f@ to @x@:
--
-- > iterate f x == [x, f x, f (f x), ...]

iterate :: (a -> a) -> a -> List a
iterate f x =  DCons x fx (iterate f (fx))
  where fx = f x

iterateFB :: (a -> b -> b) -> (a -> a) -> a -> b
iterateFB c f x = x `c` iterateFB c f (f x)


{- RULES
"iterate"    [~1] forall f x.   iterate f x = build (\c _n -> iterateFB c f x)
"iterateFB"  [1]                iterateFB (:) = iterate
 -}


-- | 'repeat' @x@ is an infinite list, with @x@ the value of every element.
repeat :: a -> List a
{-# INLINE [0] repeat #-}
-- The pragma just gives the rules more chance to fire
repeat x = xs where xs = DCons x x xs

{-# INLINE [0] repeatFB #-}     -- ditto
repeatFB :: (a -> b -> b) -> a -> b
repeatFB c x = xs where xs = x `c` xs


{- RULES
"repeat"    [~1] forall x. repeat x = build (\c _n -> repeatFB c x)
"repeatFB"  [1]  repeatFB (:)       = repeat
 -}

-- | 'replicate' @n x@ is a list of length @n@ with @x@ the value of
-- every element.
-- It is an instance of the more general 'Data.List.genericReplicate',
-- in which @n@ may be of any integral type.
{-# INLINE replicate #-}
replicate               :: Int -> a -> List a
replicate n x           =  take n (repeat x)

-- | 'cycle' ties a finite list into a circular one, or equivalently,
-- the infinite repetition of the original list.  It is the identity
-- on infinite lists.

cycle                   :: List a -> List a
cycle Nil               = error "Prelude.cycle: empty list"
cycle xs                = xs' where xs' = xs ++ xs'

-- | 'takeWhile', applied to a predicate @p@ and a list @xs@, returns the
-- longest prefix (possibly empty) of @xs@ of elements that satisfy @p@:
--
-- > takeWhile (< 3) [1,2,3,4,1,2,3,4] == [1,2]
-- > takeWhile (< 9) [1,2,3] == [1,2,3]
-- > takeWhile (< 0) [1,2,3] == []
--

takeWhile               :: (a -> Bool) -> List a -> List a
takeWhile _ Nil         =  Nil
takeWhile p (Cons x xs) 
            | p x       =  Cons x (takeWhile p xs)
            | otherwise =  Nil
takeWhile p (DCons x y xs) =
    if p x then
        if p y then
            DCons x y (takeWhile p xs)
        else
            Cons x Nil
    else
        Nil
          

-- | 'dropWhile' @p xs@ returns the suffix remaining after 'takeWhile' @p xs@:
--
-- > dropWhile (< 3) [1,2,3,4,5,1,2,3] == [3,4,5,1,2,3]
-- > dropWhile (< 9) [1,2,3] == []
-- > dropWhile (< 0) [1,2,3] == [1,2,3]
--

dropWhile               :: (a -> Bool) -> List a -> List a
dropWhile _ Nil         =  Nil
dropWhile p xs@(Cons x xs')
            | p x       =  dropWhile p xs'
            | otherwise =  xs
dropWhile p xs@(DCons x y xs') =
    if p x then
        if p y then
            dropWhile p xs'
        else
            Cons y xs'
    else
        xs

-- | 'take' @n@, applied to a list @xs@, returns the prefix of @xs@
-- of length @n@, or @xs@ itself if @n > 'length' xs@:
--
-- > take 5 "Hello World!" == "Hello"
-- > take 3 [1,2,3,4,5] == [1,2,3]
-- > take 3 [1,2] == [1,2]
-- > take 3 [] == []
-- > take (-1) [1,2] == []
-- > take 0 [1,2] == []
--
-- It is an instance of the more general 'Data.List.genericTake',
-- in which @n@ may be of any integral type.
take                   :: Int -> List a -> List a

-- | 'drop' @n xs@ returns the suffix of @xs@
-- after the first @n@ elements, or @[]@ if @n > 'length' xs@:
--
-- > drop 6 "Hello World!" == "World!"
-- > drop 3 [1,2,3,4,5] == [4,5]
-- > drop 3 [1,2] == []
-- > drop 3 [] == []
-- > drop (-1) [1,2] == [1,2]
-- > drop 0 [1,2] == [1,2]
--
-- It is an instance of the more general 'Data.List.genericDrop',
-- in which @n@ may be of any integral type.
drop                   :: Int -> List a -> List a

-- | 'splitAt' @n xs@ returns a tuple where first element is @xs@ prefix of
-- length @n@ and second element is the remainder of the list:
--
-- > splitAt 6 "Hello World!" == ("Hello ","World!")
-- > splitAt 3 [1,2,3,4,5] == ([1,2,3],[4,5])
-- > splitAt 1 [1,2,3] == ([1],[2,3])
-- > splitAt 3 [1,2,3] == ([1,2,3],[])
-- > splitAt 4 [1,2,3] == ([1,2,3],[])
-- > splitAt 0 [1,2,3] == ([],[1,2,3])
-- > splitAt (-1) [1,2,3] == ([],[1,2,3])
--
-- It is equivalent to @('take' n xs, 'drop' n xs)@.
-- 'splitAt' is an instance of the more general 'Data.List.genericSplitAt',
-- in which @n@ may be of any integral type.
splitAt                :: Int -> List a -> (List a,List a)

{- RULES
"take"     [~1] forall n xs . take n xs = takeFoldr n xs 
"takeList"  [1] forall n xs . foldr (takeFB (:) []) (takeConst []) xs n = takeUInt n xs
 -}

{- INLINE takeFoldr -}
{-
takeFoldr :: Int -> [a] -> [a]
takeFoldr (I# n#) xs
  = build (\c nil -> if n# <=# 0# then nil else
                     foldr (takeFB c nil) (takeConst nil) xs n#)
-}
{-# NOINLINE [0] takeConst #-}
-- just a version of const that doesn't get inlined too early, so we
-- can spot it in rules.  Also we need a type sig due to the unboxed Int#.
takeConst :: a -> Int# -> a
takeConst x _ = x

{-# NOINLINE [0] takeFB #-}
takeFB :: (a -> b -> b) -> b -> a -> (Int# -> b) -> Int# -> b
takeFB c n x xs m | m <=# 1#  = x `c` n
                  | otherwise = x `c` xs (m -# 1#)

{-# INLINE [0] take #-}
take (I# n#) xs = takeUInt n# xs

-- The general code for take, below, checks n <= maxInt
-- No need to check for maxInt overflow when specialised
-- at type Int or Int# since the Int must be <= maxInt

takeUInt :: Int# -> List b -> List b
takeUInt n xs
  | n >=# 0#  =  take_unsafe_UInt n xs
  | otherwise =  Nil

take_unsafe_UInt :: Int# -> List b -> List b
take_unsafe_UInt 0#  _  = Nil
take_unsafe_UInt m   ls =
  case ls of
    Nil       -> Nil
    Cons x xs -> Cons x (take_unsafe_UInt (m -# 1#) xs)
    DCons x y xs | m >=# 2#  -> DCons x y (take_unsafe_UInt (m -# 2#) xs)
                 | otherwise -> Cons x Nil

takeUInt_append :: Int# -> [b] -> [b] -> [b]
takeUInt_append n xs rs
  | n >=# 0#  =  take_unsafe_UInt_append n xs rs
  | otherwise =  []

take_unsafe_UInt_append :: Int# -> [b] -> [b] -> [b]
take_unsafe_UInt_append 0#  _ rs  = rs
take_unsafe_UInt_append m  ls rs  =
  case ls of
    []     -> rs
    (x:xs) -> x : take_unsafe_UInt_append (m -# 1#) xs rs

drop (I# n#) ls
  | n# <# 0#    = ls
  | otherwise   = drop# n# ls
    where
        drop# :: Int# -> List a -> List a
        drop# 0# xs             = xs
        drop# _  xs@Nil         = xs
        drop# m# (Cons  _   xs) = drop# (m# -# 1#) xs
        drop# m# (DCons _ x xs) 
                    | m# >=# 2# = drop# (m# -# 2#) xs
                    | otherwise = Cons x xs

splitAt (I# n#) ls
  | n# <# 0#    = (Nil, ls)
  | otherwise   = splitAt# n# ls
    where
        splitAt# :: Int# -> List a -> (List a, List a)
        splitAt# 0# xs     = (Nil, xs)
        splitAt# _  xs@Nil  = (xs, xs)
        splitAt# m# (Cons x xs) = (Cons x xs', xs'')
          where
            (xs', xs'') = splitAt# (m# -# 1#) xs
        splitAt# m# (DCons x y xs) | m# >=# 2# = (DCons x y xs',xs'')
                                   | otherwise = (Cons x Nil, Cons y xs)
          where
            (xs', xs'') = splitAt# (m# -# 2#) xs
                                       

-- | 'span', applied to a predicate @p@ and a list @xs@, returns a tuple where
-- first element is longest prefix (possibly empty) of @xs@ of elements that
-- satisfy @p@ and second element is the remainder of the list:
-- 
-- > span (< 3) [1,2,3,4,1,2,3,4] == ([1,2],[3,4,1,2,3,4])
-- > span (< 9) [1,2,3] == ([1,2,3],[])
-- > span (< 0) [1,2,3] == ([],[1,2,3])
-- 
-- 'span' @p xs@ is equivalent to @('takeWhile' p xs, 'dropWhile' p xs)@

span                    :: (a -> Bool) -> List a -> (List a,List a)
span _ xs@Nil             =  (xs, xs)
span p xs@(Cons x xs')
    | p x                 =  let (ys,zs) = span p xs' in (Cons x ys,zs)
    | otherwise           =  (Nil,xs)
span p xs@(DCons x y xs') =
    if p x then
        if p y then
            let (ys,zs) = span p xs' in (DCons x y ys,zs)
        else 
            (Cons x Nil,Cons y xs')
    else
        (Nil,xs)
           
-- | 'break', applied to a predicate @p@ and a list @xs@, returns a tuple where
-- first element is longest prefix (possibly empty) of @xs@ of elements that
-- /do not satisfy/ @p@ and second element is the remainder of the list:
-- 
-- > break (> 3) [1,2,3,4,1,2,3,4] == ([1,2,3],[4,1,2,3,4])
-- > break (< 9) [1,2,3] == ([],[1,2,3])
-- > break (> 9) [1,2,3] == ([1,2,3],[])
--
-- 'break' @p@ is equivalent to @'span' ('not' . p)@.

break                   :: (a -> Bool) -> [a] -> ([a],[a])
#ifdef USE_REPORT_PRELUDE
break p                 =  span (not . p)
#else
-- HBC version (stolen)
break _ xs@[]           =  (xs, xs)
break p xs@(x:xs')
           | p x        =  ([],xs)
           | otherwise  =  let (ys,zs) = break p xs' in (x:ys,zs)
#endif

-- | 'reverse' @xs@ returns the elements of @xs@ in reverse order.
-- @xs@ must be finite.
reverse                 :: List a -> List a
reverse l =  rev l Nil
  where
    rev Nil            a = a
    rev (Cons x xs)    a = rev xs (Cons x a)
    rev (DCons x y xs) a = rev xs (DCons y x a)

-- | 'and' returns the conjunction of a Boolean list.  For the result to be
-- 'True', the list must be finite; 'False', however, results from a 'False'
-- value at a finite index of a finite or infinite list.
and                     :: List Bool -> Bool

-- | 'or' returns the disjunction of a Boolean list.  For the result to be
-- 'False', the list must be finite; 'True', however, results from a 'True'
-- value at a finite index of a finite or infinite list.
or                      :: List Bool -> Bool
and Nil            = True
and (Cons x xs)    = x && and xs
and (DCons x y xs) = x && y && and xs
or Nil             = False
or (Cons x xs)     = x || or xs
or (DCons x y xs)  = x || y || or xs

{- RULES
"and/build"     forall (g::forall b.(Bool->b->b)->b->b) . 
                and (build g) = g (&&) True
"or/build"      forall (g::forall b.(Bool->b->b)->b->b) . 
                or (build g) = g (||) False
 -}

-- | Applied to a predicate and a list, 'any' determines if any element
-- of the list satisfies the predicate.
any                     :: (a -> Bool) -> List a -> Bool

-- | Applied to a predicate and a list, 'all' determines if all elements
-- of the list satisfy the predicate.
all                     :: (a -> Bool) -> List a -> Bool
any _ Nil            = False
any p (Cons  x xs)   = p x || any p xs
any p (DCons x y xs) = p x || p y || any p xs

all _ Nil            = True
all p (Cons  x xs)   = p x && all p xs
all p (DCons x y xs) = p x && p y && all p xs
{- RULES
"any/build"     forall p (g::forall b.(a->b->b)->b->b) . 
                any p (build g) = g ((||) . p) False
"all/build"     forall p (g::forall b.(a->b->b)->b->b) . 
                all p (build g) = g ((&&) . p) True
 -}


-- | 'elem' is the list membership predicate, usually written in infix form,
-- e.g., @x \`elem\` xs@.
elem                    :: (Eq a) => a -> List a -> Bool

-- | 'notElem' is the negation of 'elem'.
notElem                 :: (Eq a) => a -> List a -> Bool
elem _ Nil            = False
elem x (Cons  y ys)   = x==y || elem x ys
elem x (DCons y z ys) = x==y || x==z || elem x ys

notElem _ Nil            =  True
notElem x (Cons  y ys)   =  x /= y && notElem x ys
notElem x (DCons y z ys) =  x /= y && x /= z && notElem x ys
                        


-- | 'lookup' @key assocs@ looks up a key in an association list.
lookup                  :: (Eq a) => a -> List (a,b) -> Maybe b
lookup _key Nil         =  Nothing
lookup  key (Cons (x,y) xys)
    | key == x          = Just y
    | otherwise         = lookup key xys
lookup  key (DCons (x,y) (z,q) xys)
    | key == x          = Just y
    | key == z          = Just q
    | otherwise         = lookup key xys

-- | Map a function over a list and concatenate the results.
concatMap               :: (a -> List b) -> List a -> List b
concatMap f             =  foldr ((++) . f) Nil

-- | Concatenate a list of lists.
concat :: List (List a) -> List a
concat = foldr (++) Nil

{- RULES
  "concat" forall xs. concat xs = build (\c n -> foldr (\x y -> foldr c y x) n xs)
-- We don't bother to turn non-fusible applications of concat back into concat
 -}


-- | List index (subscript) operator, starting from 0.
-- It is an instance of the more general 'Data.List.genericIndex',
-- which takes an index of any integral type.
(!!)                    :: List a -> Int -> a
-- HBC version (stolen), then unboxified
-- The semantics is not quite the same for error conditions
-- in the more efficient version.
--
xs !! (I# n0) | n0 <# 0#  =  error "Prelude.(!!): negative index\n"
              | otherwise =  sub xs n0
                         where
                            sub :: List a -> Int# -> a
                            sub Nil     _ = error "Prelude.(!!): index too large\n"
                            sub (Cons y ys) n = 
                                if n ==# 0#
                                then y
                                else sub ys (n -# 1#)
                            sub (DCons x y ys) n = 
                                case n of
                                  0# -> x
                                  1# -> y
                                  _  -> sub ys (n -# 2#)

{-

%*********************************************************
%*                                                      *
\subsection{The zip family}
%*                                                      *
%*********************************************************

\begin{code}
-}
foldr2 :: (a -> b -> c -> c) -> c -> [a] -> [b] -> c
foldr2 _k z []    _ys    = z
foldr2 _k z _xs   []     = z
foldr2 k z (x:xs) (y:ys) = k x y (foldr2 k z xs ys)

foldr2_left :: (a -> b -> c -> d) -> d -> a -> ([b] -> c) -> [b] -> d
foldr2_left _k  z _x _r []     = z
foldr2_left  k _z  x  r (y:ys) = k x y (r ys)

foldr2_right :: (a -> b -> c -> d) -> d -> b -> ([a] -> c) -> [a] -> d
foldr2_right _k z  _y _r []     = z
foldr2_right  k _z  y  r (x:xs) = k x y (r xs)

-- foldr2 k z xs ys = foldr (foldr2_left k z)  (\_ -> z) xs ys
-- foldr2 k z xs ys = foldr (foldr2_right k z) (\_ -> z) ys xs
{- RULES
"foldr2/left"   forall k z ys (g::forall b.(a->b->b)->b->b) . 
                  foldr2 k z (build g) ys = g (foldr2_left  k z) (\_ -> z) ys

"foldr2/right"  forall k z xs (g::forall b.(a->b->b)->b->b) . 
                  foldr2 k z xs (build g) = g (foldr2_right k z) (\_ -> z) xs
 -}
{-

\end{code}

The foldr2/right rule isn't exactly right, because it changes
the strictness of foldr2 (and thereby zip)

E.g. main = print (null (zip nonobviousNil (build undefined)))
          where   nonobviousNil = f 3
                  f n = if n == 0 then [] else f (n-1)

I'm going to leave it though.


Zips for larger tuples are in the List module.

\begin{code}
-}
----------------------------------------------
-- | 'zip' takes two lists and returns a list of corresponding pairs.
-- If one input list is short, excess elements of the longer list are
-- discarded.
zip :: List a -> List b -> List (a,b)
zip (Cons  a   as) (Cons  b   bs) = Cons  (a,b) (zip as bs)
zip (Cons  a   as) (DCons b y bs) = Cons  (a,b) (zip as (Cons y bs))
zip (DCons a x as) (Cons  b   bs) = Cons  (a,b) (zip (Cons x as) bs)
zip (DCons a x as) (DCons b y bs) = DCons (a,b) (x,y) (zip as bs)
zip _      _                      = Nil

{-# INLINE [0] zipFB #-}
zipFB :: ((a, b) -> c -> d) -> a -> b -> c -> d
zipFB c x y r = (x,y) `c` r

{- RULES
"zip"      [~1] forall xs ys. zip xs ys = build (\c n -> foldr2 (zipFB c) n xs ys)
"zipList"  [1]  foldr2 (zipFB (:)) []   = zip
 -}



----------------------------------------------
-- | 'zip3' takes three lists and returns a list of triples, analogous to
-- 'zip'.
zip3 :: [a] -> [b] -> [c] -> [(a,b,c)]
-- Specification
-- zip3 =  zipWith3 (,,)
zip3 (a:as) (b:bs) (c:cs) = (a,b,c) : zip3 as bs cs
zip3 _      _      _      = []



-- The zipWith family generalises the zip family by zipping with the
-- function given as the first argument, instead of a tupling function.


----------------------------------------------
-- | 'zipWith' generalises 'zip' by zipping with the function given
-- as the first argument, instead of a tupling function.
-- For example, @'zipWith' (+)@ is applied to two lists to produce the
-- list of corresponding sums.
zipWith :: (a->b->c) -> [a]->[b]->[c]
zipWith f (a:as) (b:bs) = f a b : zipWith f as bs
zipWith _ _      _      = []

{-# INLINE [0] zipWithFB #-}
zipWithFB :: (a -> b -> c) -> (d -> e -> a) -> d -> e -> b -> c
zipWithFB c f x y r = (x `f` y) `c` r

{- RULES
"zipWith"       [~1] forall f xs ys.    zipWith f xs ys = build (\c n -> foldr2 (zipWithFB c f) n xs ys)
"zipWithList"   [1]  forall f.  foldr2 (zipWithFB (:) f) [] = zipWith f
  -}



-- | The 'zipWith3' function takes a function which combines three
-- elements, as well as three lists and returns a list of their point-wise
-- combination, analogous to 'zipWith'.
zipWith3                :: (a->b->c->d) -> [a]->[b]->[c]->[d]
zipWith3 z (a:as) (b:bs) (c:cs)
                        =  z a b c : zipWith3 z as bs cs
zipWith3 _ _ _ _        =  []

-- | 'unzip' transforms a list of pairs into a list of first components
-- and a list of second components.
unzip    :: List (a,b) -> (List a,List b)
{-# INLINE unzip #-}
unzip    =  foldr (\(a,b) ~(as,bs) -> (Cons a as,Cons b bs)) (Nil,Nil)

-- | The 'unzip3' function takes a list of triples and returns three
-- lists, analogous to 'unzip'.
unzip3   :: List (a,b,c) -> (List a,List b,List c)
{-# INLINE unzip3 #-}
unzip3   =  foldr (\(a,b,c) ~(as,bs,cs) -> (Cons a as,Cons b bs,Cons c cs))
                  (Nil,Nil,Nil)
{-


%*********************************************************
%*                                                      *
\subsection{Error code}
%*                                                      *
%*********************************************************

Common up near identical calls to `error' to reduce the number
constant strings created when compiled:

-}

errorEmptyList :: String -> a
errorEmptyList fun =
  error (prel_list_str P.++ fun P.++ ": empty list")

prel_list_str :: String
prel_list_str = "Prelude."


{- Here begins the stuff from GHC.Base -}


-- do explicitly: deriving (Eq, Ord)
-- to avoid weird names like con2tag_[]#

instance (Eq a) => Eq (List a) where
    {-# SPECIALISE instance Eq (List Char) #-}
    Nil            == Nil            = True
    (Cons x xs)    == (Cons y ys)    = x == y && xs == ys
    (DCons x z xs) == (DCons y q ys) = x == y && z == q && xs == ys
    (Cons x xs)    == (DCons y q ys) = x == y && xs == (Cons q ys)
    (DCons x z xs) == (Cons y ys)    = x == y && (Cons x xs) == ys
    _xs    == _ys    = False

instance (Ord a) => Ord (List a) where
    {-# SPECIALISE instance Ord (List Char) #-}
    compare Nil            Nil            = EQ
    compare Nil            (Cons _ _)     = LT
    compare Nil            (DCons _ _ _)  = LT
    compare (Cons _ _)     Nil            = GT
    compare (DCons _ _ _)  Nil            = GT
    compare (Cons x xs)    (Cons y ys)    = case compare x y of
                                              EQ    -> compare xs ys
                                              other -> other
    compare (DCons x z xs) (DCons y q ys) = case compare x y of
                                              EQ    -> case compare z q of
                                                         EQ    -> compare xs ys
                                                         other -> other
                                              other -> other
    compare (Cons x xs)    (DCons y q ys) = case compare x y of
                                              EQ    -> compare xs (Cons q ys) 
                                              other -> other
    compare (DCons x z xs) (Cons y ys)    = case compare x y of
                                              EQ    -> compare (Cons x xs) ys
                                              other -> other


instance Functor List where
    fmap = map

instance  Monad List  where
    m >>= k             = foldr ((++) . k) Nil m
    m >> k              = foldr ((++) . (\ _ -> k)) Nil m
    return x            = Cons x Nil
    fail _              = Nil
{-

A few list functions that appear here because they are used here.
The rest of the prelude list functions are in GHC.List.

----------------------------------------------
--      foldr/build/augment
----------------------------------------------
  
-}
-- | 'foldr', applied to a binary operator, a starting value (typically
-- the right-identity of the operator), and a list, reduces the list
-- using the binary operator, from right to left:
--
-- > foldr f z [x1, x2, ..., xn] == x1 `f` (x2 `f` ... (xn `f` z)...)

foldr            :: (a -> b -> b) -> b -> List a -> b
-- foldr _ z []     =  z
-- foldr f z (x:xs) =  f x (foldr f z xs)
{-# INLINE [0] foldr #-}
-- Inline only in the final stage, after the foldr/cons rule has had a chance
foldr k z xs = go xs
             where
               go Nil            = z
               go (Cons y    ys) = y `k` go ys
               go (DCons x y ys) = k x (k y (go ys))

-- | A list producer that can be fused with 'foldr'.
-- This function is merely
--
-- >    build g = g (:) []
--
-- but GHC's simplifier will transform an expression of the form
-- @'foldr' k z ('build' g)@, which may arise after inlining, to @g k z@,
-- which avoids producing an intermediate list.
{-
build   :: forall a. (forall b. (a -> b -> b) -> b -> b) -> [a]
{-# INLINE [1] build #-}
        -- The INLINE is important, even though build is tiny,
        -- because it prevents [] getting inlined in the version that
        -- appears in the interface file.  If [] *is* inlined, it
        -- won't match with [] appearing in rules in an importing module.
        --
        -- The "1" says to inline in phase 1

build g = g (:) []
-}
-- | A list producer that can be fused with 'foldr'.
-- This function is merely
--
-- >    augment g xs = g (:) xs
--
-- but GHC's simplifier will transform an expression of the form
-- @'foldr' k z ('augment' g xs)@, which may arise after inlining, to
-- @g k ('foldr' k z xs)@, which avoids producing an intermediate list.
{-
augment :: forall a. (forall b. (a->b->b) -> b -> b) -> [a] -> [a]
{-# INLINE [1] augment #-}
augment g xs = g (:) xs

{-# RULES
"fold/build"    forall k z (g::forall b. (a->b->b) -> b -> b) . 
                foldr k z (build g) = g k z

"foldr/augment" forall k z xs (g::forall b. (a->b->b) -> b -> b) . 
                foldr k z (augment g xs) = g k (foldr k z xs)

"foldr/id"                        foldr (:) [] = \x  -> x
"foldr/app"     [1] forall ys. foldr (:) ys = \xs -> xs ++ ys
        -- Only activate this from phase 1, because that's
        -- when we disable the rule that expands (++) into foldr

-- The foldr/cons rule looks nice, but it can give disastrously
-- bloated code when commpiling
--      array (a,b) [(1,2), (2,2), (3,2), ...very long list... ]
-- i.e. when there are very very long literal lists
-- So I've disabled it for now. We could have special cases
-- for short lists, I suppose.
-- "foldr/cons" forall k z x xs. foldr k z (x:xs) = k x (foldr k z xs)

"foldr/single"  forall k z x. foldr k z [x] = k x z
"foldr/nil"     forall k z.   foldr k z []  = z 

"augment/build" forall (g::forall b. (a->b->b) -> b -> b)
                       (h::forall b. (a->b->b) -> b -> b) .
                       augment g (build h) = build (\c n -> g c (h c n))
"augment/nil"   forall (g::forall b. (a->b->b) -> b -> b) .
                        augment g [] = build g
 #-}
-}
-- This rule is true, but not (I think) useful:
--      augment g (augment h t) = augment (\cn -> g c (h c n)) t


----------------------------------------------
--              map     
----------------------------------------------

-- | 'map' @f xs@ is the list obtained by applying @f@ to each element
-- of @xs@, i.e.,
--
-- > map f [x1, x2, ..., xn] == [f x1, f x2, ..., f xn]
-- > map f [x1, x2, ...] == [f x1, f x2, ...]

map :: (a -> b) -> List a -> List b
map _ Nil            = Nil
map f (Cons x xs)    = Cons  (f x)       (map f xs)
map f (DCons x y xs) = DCons (f x) (f y) (map f xs)

-- Realigning version
mapR :: (a -> b) -> List a -> List b
mapR _ Nil            = Nil
mapR f (Cons x xs)    = mapR1 f (f x) xs
mapR f (DCons x y xs) = DCons (f x) (f y) (mapR f xs)

mapR1 _ fx Nil            = Cons fx Nil
mapR1 f fx (Cons x    xs) = DCons fx (f x) (mapR f xs)
mapR1 f fx (DCons x y xs) = DCons fx (f x) (mapR1 f (f y) xs)

-- Note eta expanded
mapFB ::  (elt -> lst -> lst) -> (a -> elt) -> a -> lst -> lst
{-# INLINE [0] mapFB #-}
mapFB c f x ys = c (f x) ys

-- The rules for map work like this.
-- 
-- Up to (but not including) phase 1, we use the "map" rule to
-- rewrite all saturated applications of map with its build/fold 
-- form, hoping for fusion to happen.
-- In phase 1 and 0, we switch off that rule, inline build, and
-- switch on the "mapList" rule, which rewrites the foldr/mapFB
-- thing back into plain map.  
--
-- It's important that these two rules aren't both active at once 
-- (along with build's unfolding) else we'd get an infinite loop 
-- in the rules.  Hence the activation control below.
--
-- The "mapFB" rule optimises compositions of map.
--
-- This same pattern is followed by many other functions: 
-- e.g. append, filter, iterate, repeat, etc.

{- RULES
"map"       [~1] forall f xs.   map f xs                = build (\c n -> foldr (mapFB c f) n xs)
"mapList"   [1]  forall f.      foldr (mapFB (:) f) []  = map f
"mapFB"     forall c f g.       mapFB (mapFB c f) g     = mapFB c (f.g) 
  -}



----------------------------------------------
--              append  
----------------------------------------------

-- | Append two lists, i.e.,
--
-- > [x1, ..., xm] ++ [y1, ..., yn] == [x1, ..., xm, y1, ..., yn]
-- > [x1, ..., xm] ++ [y1, ...] == [x1, ..., xm, y1, ...]
--
-- If the first list is not finite, the result is the first list.

(++) :: List a -> List a -> List a
(++) Nil            ys = ys
(++) (Cons x xs)    ys = Cons x (xs ++ ys)
(++) (DCons x y xs) ys = DCons x y (xs ++ ys)

{- RULES
"++"    [~1] forall xs ys. xs ++ ys = augment (\c n -> foldr c n xs) ys
  -}

-- Enumeration
-- This doesn't come from the list modules but from the enum module
-- I put it here anyway to be able to give optimized versions

-- Default implementations from GHC.Enum. Requires the instance Enum Int.

enumFrom :: Enum a => a -> List a
enumFrom x = map toEnum (enumFromInt (fromEnum x))

enumFromTo :: Enum a => a -> a -> List a
enumFromTo x y = map toEnum (enumFromToInt (fromEnum x) (fromEnum y))

enumFromThen :: Enum a => a -> a -> List a
enumFromThen x y = map toEnum (enumFromThenInt (fromEnum x) (fromEnum y))

enumFromThenTo :: Enum a => a -> a -> a -> List a
enumFromThenTo x1 x2 y = map toEnum (enumFromThenToInt (fromEnum x1) (fromEnum x2) (fromEnum y))

-- Int versions of the enum functions

enumFromInt :: Int -> List Int
enumFromInt (I# a) = efInt a
efInt a = DCons (I# a) (I# (a +# 1#)) (efInt (a +# 2#))

enumFromThenInt :: Int -> Int -> List Int
enumFromThenInt (I# a) (I# b) = efthInt a (b -# a)
efthInt a d = DCons (I# a) (I# (a +# d)) (efthInt a (a +# d +# d))

{- Old version. It has too complicated comparisons in the inner loop. -}
enumFromToInt2 :: Int -> Int -> List Int
enumFromToInt2 (I# a) (I# b) = eftInt2 a b
eftInt2 a b
    | a ># b          = Nil
    | (a +# 1#) <=# b = DCons (I# a) (I# (a +# 1#)) (eftInt2 (a +# 2#) b)
    | otherwise       = Cons  (I# a) Nil

-- Faster version
enumFromToInt (I# a) (I# b) = eftInt a b
eftInt a b 
    | a ># b = Nil
    | (b -# a) `modInt#` 2# ==# 0# = Cons (I# a) (eftIntWork (a +# 1#) b)
    | otherwise = eftIntWork a b
-- Assumes there's an even number of ints to be produced.
eftIntWork a b
    | a ># b    = Nil
    | otherwise = DCons (I# a) (I# (a +# 1#)) (eftIntWork (a +# 2#) b)

enumFromThenToInt :: Int -> Int -> Int -> List Int
enumFromThenToInt (I# a) (I# b) (I# c) = efttInt a (b -# a) c
efttInt a d c
    | a ># c         = Nil
    | (a +# d) <=# c = DCons (I# a) (I# (a +# d)) (efttInt (a +# d +# d) d c)
    | otherwise      = Cons  (I# a) Nil
