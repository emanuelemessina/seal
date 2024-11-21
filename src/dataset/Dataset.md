# Dataset Notes

## Building the character set

1. The Kangxi dictionary contains a plethora of traditional characters (40k+), only those in the Unicode CJK range (20k+) were extracted: we call this the "CJK Standalone" set (because these are the unified traditional characters that contain also some simplified characters that retain semantic differences in classical chinese, thus they should not count as merely simplified variants, instead all the true simplified variants are removed) yielding 16k+ characters.
2. A bunch of free fonts for seal script was collected from the web, and for each character in the standalone set we associate every font that supports it (by calling either the traditional or the simplified code)
3. We prune the standalone set, eliminating the characters that are not representable with any of the available fonts, yielding 13k+ characters.