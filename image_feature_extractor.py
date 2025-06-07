from skimage.feature.texture import graycomatrix, graycoprops
from skimage.feature import local_binary_pattern, hog
from skimage.color import rgb2hsv
import numpy as np
import cv2

class ImageFeatureExtractor:
    
    @staticmethod
    def extract_all_descriptors(img_bgr):
        """
        Dada una imagen en BGR (OpenCV) redimensionada a (128×192) o (128×128),
        calcula y concatena:
        [HSV_512, LBP_26, GLCM_24/48, HOG_1764, HuMoments_7].
        Ajusta la longitud de cada vector según tu configuración.
        Retorna un vector 1D de features.
        """
        # 1) Histograma HSV 8×8×8 → 512
        hist_hsv = ImageFeatureExtractor.extract_hsv_histogram(img_bgr, bins=(8,8,8))

        # 2) LBP (26 bins)
        # Nota: puedes pasar img_bgr original (no redimensionado) o redimensionar primero.
        lbp = ImageFeatureExtractor.extract_lbp(img_bgr, numPoints=24, radius=8)

        # 3) GLCM multiescala/ángulo: props * distancias * ángulos
        glcm_feats = ImageFeatureExtractor.extract_glcm(img_bgr, distances=[1,2], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4])
        # si distances=[1,2] y angles=[0,π/4,π/2,3π/4], dim GLCM = 6*2*4 = 48

        # 4) HOG (1764)
        hog_feats = ImageFeatureExtractor.extract_hog(img_bgr, resize_dim=(128,128))

        # 5) Momentos de Hu (7)
        hu_feats = ImageFeatureExtractor.extract_hu_moments(img_bgr, resize_dim=(128,128))

        # Concatenar todo
        return np.hstack([hist_hsv, lbp, glcm_feats, hog_feats, hu_feats])

    @staticmethod
    def extract_hsv_histogram(img_bgr, bins=(8,8,8)):
        """
        img_bgr: imagen en BGR (OpenCV), tamaño arbitrario.
        bins: número de celdas por canal H, S y V.
        Retorna: vector de length = bins[0] * bins[1] * bins[2], normalizado (suma = 1).
        """
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([img_hsv], [0,1,2], None, bins, [0,180, 0,256, 0,256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist  # e.g., length 8*8*8 = 512

    @staticmethod
    def extract_lbp(img_bgr, numPoints=24, radius=8):
        """
        Extrae LBP uniform con numPoints y radius sobre la imagen en escala de grises.
        Retorna histograma normalizado de length = numPoints + 2 (26).
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        lbp  = local_binary_pattern(gray, numPoints, radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                bins=np.arange(0, numPoints + 3),
                                range=(0, numPoints + 2))
        hist = hist.astype("float") / (hist.sum() + 1e-7)
        return hist  # length = 26

    @staticmethod
    def extract_glcm(img_bgr, distances=[1,2], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
        """
        Extrae propiedades Haralick de la matriz GLCM para múltiples distancias y ángulos.
        Retorna un vector [contrast, dissimilarity, homogeneity, energy, correlation, ASM] 
        concatenado por cada par (distancia, ángulo).
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        glcm = graycomatrix(gray,
                            distances=distances,
                            angles=angles,
                            levels=256,
                            symmetric=True,
                            normed=True)
        props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
        feats = []
        for p in props:
            # graycoprops devuelve la matriz de tamaño (len(distances), len(angles))
            vals = graycoprops(glcm, p)
            feats.append(vals.flatten())
        return np.hstack(feats)  # length = 6 * len(distances) * len(angles)

    @staticmethod
    def extract_hog(img_bgr, resize_dim=(128,128)):
        """
        Extrae HOG de la versión redimensionada a resize_dim.
        Retorna vector HOG normalizado.
        """
        img_resized = cv2.resize(img_bgr, resize_dim)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        hog_feat, _ = hog(
            gray,
            pixels_per_cell=(16,16),
            cells_per_block=(2,2),
            orientations=9,
            visualize=True,
            feature_vector=True
        )
        return hog_feat  # length ≈ 1764 para 128×128

    @staticmethod
    def extract_hu_moments(img_bgr, resize_dim=(128,128)):
        """
        Extrae los 7 momentos invariantes de Hu de la imagen redimensionada.
        Retorna un vector de length=7.
        """
        img_resized = cv2.resize(img_bgr, resize_dim)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        moments = cv2.moments(gray)
        hu = cv2.HuMoments(moments).flatten()
        # Opcional: aplicar log escalar (log transform) para comprimir rango
        hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-7)
        return hu  # length = 7