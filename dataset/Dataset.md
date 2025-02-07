# Dataset Notes

## Building the character set

1. The Kangxi dictionary contains a plethora of traditional characters (40k+), only those in the Unicode CJK range (20k+) were extracted: we call this the "CJK Standalone" set (because these are the unified traditional characters that contain also some simplified characters that retain semantic differences in classical chinese, thus they should not count as merely simplified variants, instead all the true simplified variants are removed) yielding 16k+ characters.
2. A bunch of free fonts for seal script was collected from the web, and for each character in the standalone set we associate every font that supports it (by calling either the traditional or the simplified code)
3. We prune the standalone set, eliminating the characters that are not representable with any of the available fonts, yielding 13k+ characters.
4. At this point, there are many characters (5k+, ~40%) that are represented by only one font among the collected ones. This is no coincidence, in fact they correspond to very rare characters that only the most complete fonts can code for. Considering that their presence would lead to an imbalanced dataset as well as a uslessly big model, we eliminate them. We are left with 8k+ characters.
5. Repeating the process to further simplify the problem, we drop each character supported by with less than half of the available fonts, as they contributed for less than half of the dataset (we relax our metric for "rare" characters). We are left with 5k+ characters, each represented by at least 10 fonts.
6. For our preliminary tests we reduced the dataset even more, taking only the first 1k/3k characters from a frequency list based on classical chinese corpus.