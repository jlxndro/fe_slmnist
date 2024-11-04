import sys
import os

### Paths to dependencies -----------------------------------------------------------

# AWS
pathBotocore = "D:\PythonDependencies\Botocore"
pathBoto3 = "D:\PythonDependencies\Boto3"

# Image Processing Tools
pathOpenCV = "D:\PythonDependencies\OpenCV"
pathEasyOCR = "D:\PythonDependencies\EasyOCR"
pathUltralytics = "D:/PythonDependencies/Ultralytics"
pathScikitImage = "D:\PythonDependencies\Scikit-image"

# Visualisation
pathMatplotlib = "D:\PythonDependencies\Matplotlib"

# Algorithms
pathFilterpy = "D:\PythonDependencies\Filterpy"

# Web-Framework
pathStreamlit = "D:\PythonDependencies\Streamlit"
pathStreamlitBabel = "D:\PythonDependencies\Streamlit-Babel"
pathTensorflow = "D:\PythonDependencies\Tensorflow"

### Adding paths to the system path --------------------------------------------------

sys.path.append(pathBotocore)
sys.path.append(pathBoto3)

sys.path.append(pathOpenCV)
sys.path.append(pathEasyOCR)
sys.path.append(pathUltralytics)
sys.path.append(pathScikitImage)

sys.path.append(pathMatplotlib)

sys.path.append(pathFilterpy)

sys.path.append(pathStreamlit)
sys.path.append(pathStreamlitBabel)
sys.path.append(pathTensorflow)