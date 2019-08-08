# LineScore
A feature extraction tool to detect line/fiber-like structures in an image.


In order to quantify the relative abundance of filaments between the controls and the perturbed replicates, we developed a filament analysis tool. The tool employs a 7x7 pixel window (can be modified), which sweeps over the image, and assigns a “Line Score” to the central pixel of the window. The line score acts as a measure of probability, that the focal pixel is part of a curvilinear structure, such as a keratin filament. Pixels located on line-like structures (such as keratin fibres) yield a high line score, whereas other pixels yield low line scores. Therefore, the mean line score obtained from an image acts as a measure of the abundance of filamentous structures in the image. 


# How to Use

To run the script on *filename*.jpg, follow these steps:
  
1. Using ImageJ, FIJI or any other image processing software, create a mask of the area that must be analyzed. The area to be analyzed must be white, whereas everything else must be black. Rename this image *filename*_highcontrast.jpg
  
2. Open terminal and cd to directory :
   python run_script.py *filename*
  
 
 
