# Stress316L: Dataset of Strain & Stress of Stainless Steel 316L during Cold Tension. 

*******************************************************
* Stress316L: Dataset of Strain & Stress of Stainles  *
*           Steel 316L during Cold Tension            *
*                                                     *
*                                                     *
*             TRAINING, VALIDATION, TEST SETS         *
*                                                     *
*                                                     *
* https://www.kaggle.com/mahshadlotfinia/Stress316L   *
* mahshad.lotfinia@alum.sharif.edu                    *
*                                                     *
*******************************************************

Version 1.0: Februrary 2, 2021


In case you use this dataset, please cite the original paper:


Mahshad Lotfinia, and Soroosh Tayebi Arasteh. "Machine Learning-Based Generalized Model for Finite Element Analysis of Roll Deflection During the Austenitic Stainless Steel 316L Strip Rolling". ArXive eprint, February 2021.

BibTex
	@article{Stress316L,
	  title = "How Machine Learning-Based Generalized Model for Finite Element Analysis of Roll Deflection During the Austenitic Stainless Steel 316L Strip Rolling",
	  author = "Lotfinia, Mahshad and Tayebi Arasteh, Soroosh",
	  journal = "Arxive preprint",
	  url = "https://www.github.com/mahshadlotfinia/Stress316L/",
	  month = "02",       
      year = "2021"
	 }


LIST OF VERSIONS

  v1.0 [02.02.2021]: initial distribution of the data

=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
SUMMARY

Unlike the other groups of metals, Austenitic Stainless Steel 316L has an unpredictable Strain-Stress curve. 
Thus, we conducted a series of mechanical tensile tests at different strain rates.
Afterwards, using this dataset we can train a neural network to predict the best 
Strain-Stress curve that predicts more accurate values of the flow stress during the cold deformation.

=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
DATA COLLECTION

We conducted four sets of Uniaxial Tensile Tests in 0.001S−1, 0.00052S−1, 0.0052S−1, and 0.052S−1 
strain rates in the room temperature on our Austenitic Stainless Steel 316L sample. According to the ASTME8 standard, 
the ASS316L sheets with an initial thickness of 4 mm, width of 6 mm, and Gage length of 32 mm 
were utilized for thetensile tests using a compression test machine (Electro Mechanic Instron 4208). 
The results were transferred to the Santam Machine Controller software for recording, which led 
to obtaining the extension data (in mm) and the force data (in N), which were converted to 
the true-strain and true-stress values. The data conversion procedure was done by considering 
the cross-section of the loaded force, which for our case was 24 mm^2.

=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
DATASET CONTENTS

15,858 different Strain-Stress values at 4 different strain rates.

./Stress316L_data/labels.csv: Stress values.
./Stress316L_data/features.csv: Strain & Strain rate values for the corresponding points in the ./Stress316L_data/labels.csv.
./Stress316L_data/x_y_initial.csv: Strain-Stress values.
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
DATA FORMAT FOR ALL THE FILES


All the files are provided in the "csv" format.

=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
The dataset URL:

	https://kaggle.com/mahshadlotfinia/Stress316L/

=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
LICENSE


The accompanying dataset is released under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).


=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
SOURCE CODE


The official source code of the paper: https://github.com/mahshadlotfinia/316lfemann/

=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
CONTACT

E-mail: mahshad.lotfinia@alum.sharif.edu


REFERENCES:


Materials Science and Engineering Mechanical Lab, the Sharif University of Technology, Tehran, Iran.


