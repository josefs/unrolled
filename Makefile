.PHONY : bench

GHC = ghc -O2 -c -fext-core

bench:	Time.o
	ghc -e main Time

Data/List/Unrolled3.hs: Generate.hs
	ghc -e "writeModules 4" Generate.hs

Data/List/Unrolled4.hs: Generate.hs
	ghc -e "writeModules 4" Generate.hs

Time.o: Time.hs Data/List/Unrolled3.o Data/List/Unrolled4.o Data/List/Unrolled.o
	$(GHC) Time.hs

Data/List/Unrolled3.o:Data/List/Unrolled3.hs
	$(GHC) Data/List/Unrolled3.hs

Data/List/Unrolled4.o:Data/List/Unrolled4.hs
	$(GHC) Data/List/Unrolled4.hs

Data/List/Unrolled.o:Data/List/Unrolled.hs
	$(GHC) Data/List/Unrolled.hs
