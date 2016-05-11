DenseCut ported to Linux (now cross-platform between Windows and Linux)

Uses Code::Blocks for compiliation on Linux.
1. Open and compile "CmCode" and then "mmcheng_densecut"
2. Open terminal, cd into the "mmcheng_densecut" folder, and run the command "./bin/mmcheng_densecut ./"
3. Lots of results will be written to "mmcheng_densecut/ASD/Sal4N" which you can compare to the paper results in "Paper_Claimed_Results" which were copied from the PDF.

To test on new images, save to "ASD/Imgs" with ".jpg" extension. Corresponding rectangular boxes must be saved as grayscale images with ".png" extension (with the same base file name). The rectangular boxes must be thicker than 1 pixel or else the program will crash.

See M.M. Cheng's website for license info and updates:
http://mmcheng.net/densecut/

The "CmCode" folder portion of this repository comes from:
https://github.com/MingMingCheng/CmCode
