import sys

import numpy as np
import SimpleITK as sitk
from skimage import filters, measure

import random
from math import pi
from scipy.stats import truncnorm, uniform

from .utils import get_study_uid
from library_dicom.dicom_processor.model.Fusion import Fusion 
from library_dicom.dicom_processor.model.FusionMask import FusionMask 

"""Classes for pre-processing : read NIFTI, threshold mask, reshape PET/CT/mask, 
    scale intensity, concatenate PET/CT ... 


    """

class Compose(object):
    """A class to applied a list of transformations on dict dataset

    """
    def __init__(self, transformers):
        self.transformers = transformers

    def __call__(self, img_dict):

        for transform in self.transformers:
            img_dict = transform(img_dict)
        return img_dict


class LoadNifti(object):
    """
    Load Nifti images and returns Simple itk image
    """
    #enlever valeur par defaut de keys
    def __init__(self, keys,
                 dtypes=None,
                 image_only=False):
        self.keys = (keys,) if isinstance(keys, str) else keys

        if dtypes is None:
            
            dtypes = {'pet_img': sitk.sitkFloat32,
                      'ct_img': sitk.sitkFloat32,
                      'mask_img': sitk.sitkUInt8}
        self.keys = keys
        self.image_only = image_only

        self.dtypes = dtypes
    def __call__(self, img_dict):
        output = dict()
        output['image_id'] = get_study_uid(img_dict[self.keys[0]])
        for key in img_dict.keys():
            # check img_dict[key] == str
            output[key] = sitk.ReadImage(img_dict[key], self.dtypes[key])
            if self.image_only:
                output[key] = sitk.GetArrayFromImage(output[key])

        return output


class LoadNumpy(object):
    """
    Load .npy files and return ndimage
    """

    def __init__(self, keys):
        self.keys = (keys,) if isinstance(keys, str) else keys

    def __call__(self, img_dict):
        for key in self.keys:
            img_dict[key] = np.load(img_dict[key])

        return img_dict

class ResampleReshapeAlign(object):
    """1) resample, reshape align PT and CT 
       2) resample reshape align MASK and PT 

    Args:
        object ([type]): [description]
    """

    #import Fusion and FusionMask from DicomToCNN
    def __init__(self,target_size, target_spacing, target_direction, target_origin=None, keys=("pet_img", "ct_img", "mask_img"),mask=True):
        self.keys = (keys,) if isinstance(keys, str) else keys
        self.target_size = target_size
        self.target_spacing = target_spacing
        self.target_direction = target_direction
        self.target_origin = target_origin
        self.mask=mask

    def __call__(self, img_dict):
        #1
        fusion_object = Fusion(img_dict[self.keys[0]], img_dict[self.keys[1]], self.target_size, self.target_spacing, self.target_direction, mode ='dict') 
        img_dict[self.keys[0]], img_dict[self.keys[1]] = fusion_object.resample(mode='head')

        #rajout condition mask     
        if self.mask: 
            #2
            fusion_mask_object = FusionMask(img_dict[self.keys[0]], img_dict[self.keys[2]], self.target_size, self.target_spacing, self.target_direction, mode ='dict')
            img_dict[self.keys[2]] = fusion_mask_object.resample()

        return img_dict 

"""
class ResampleMask(object):


    #import fonction from Dicom-To-CNN : resample_mask_nifti
    def __init__(self, keys=('merged_img', 'mask_img'),target_size, target_spacing, target_direction, target_origin = None):
        self.keys = (keys,) if isinstance(keys, str) else keys
        self.target_size = target_size
        self.target_spacing = target_spacing
        self.target_direction = target_direction
        self.target_origin = targer_origin

    def __call__(self, img_dict):
        img_dict[self.keys[1]] = resample_mask_nifti(img_dict[self.keys[0]], img_dict[self.keys[1]], self.target_size, self.target_spacing, self.target_direction, self.target_origin)
        return img_dict
""" 



class Roi2Mask(object):
    """
    Apply threshold-based method to determine the segmentation from the ROI
    """

    def __init__(self, keys=('pet_img', 'mask_img'), method='otsu', tval=0.0, new_key_name=None):
        """
        :param keys: 'pet_img' must be a 3D simpleITK image
                     'mask_img' must be a 4D simpleITK image. shape = (n_roi, _, _, _)
        :param method: method to use for calculate the threshold
                Must be one of 'absolute', 'relative', 'otsu', 'adaptative'
        :param tval: Used only for method= 'absolute' or 'relative'. threshold value of the method.
                for 2.5 SUV threshold: use method='absolute', tval=2.5
                for 41% SUV max threshold: method='relative', tval=0.41
        """
        self.keys = (keys,) if isinstance(keys, str) else keys

        self.keys = keys
        self.method = method.lower()
        self.tval = tval
        self.new_key_name = new_key_name if new_key_name is not None else keys[1]

        assert method in ['absolute', 'relative', 'otsu', 'adaptative']

    def __call__(self, img_dict):
        pet_key = self.keys[0]
        roi_key = self.keys[1]

        img_dict[self.new_key_name] = self.roi2mask(img_dict[roi_key], img_dict[pet_key])
        return img_dict

    def calculate_threshold(self, roi):
        """[summary]

        Args:
            roi ([ndarray]): [ROI array]

        Returns:
            [float]: [threshold value]
        """
        if self.method == 'absolute':
            return self.tval

        elif self.method == 'relative':
            # check len(roi) > 0
            SUV_max = np.max(roi)
            return self.tval * SUV_max

        elif self.method == 'otsu_abs' or self.method == 'otsu':
            # check len(np.unique(roi)) > 1
            return filters.threshold_otsu(roi)

    def roi2mask(self, mask_img, pet_img):
        """
        Generate the mask from the ROI of the pet scan
        Args:
            :param mask_img: sitk image, raw mask (i.e ROI)
            :param pet_img: sitk image, the corresponding pet scan

        :return: sitk image, the ground truth segmentation
        """
        # transform to numpy
        mask_array = sitk.GetArrayFromImage(mask_img)
        pet_array = sitk.GetArrayFromImage(pet_img)

        # get 3D meta information
        if len(mask_array.shape) == 3:
            mask_array = np.expand_dims(mask_array, axis=0)

            origin = mask_img.GetOrigin()
            spacing = mask_img.GetSpacing()
            direction = tuple(mask_img.GetDirection())
            # size = mask_img.GetSize()
        else:

            # convert false-4d meta information to 3d information
            origin = mask_img.GetOrigin()[:-1]
            spacing = mask_img.GetSpacing()[:-1]
            direction = tuple(el for i, el in enumerate(mask_img.GetDirection()[:12]) if not (i + 1) % 4 == 0)
            # size = mask_img.GetSize()[:-1]

        new_mask = np.zeros(mask_array.shape[1:], dtype=np.int8)

        for num_slice in range(mask_array.shape[0]):
            mask_slice = mask_array[num_slice] #ROI 3D MATRIX
            roi = pet_array[mask_slice > 0]
            if len(roi) == 0:
                # R.O.I is empty
                continue
            try:
                threshold = self.calculate_threshold(roi)

                # apply threshold
                new_mask[np.where((pet_array >= threshold) & (mask_slice > 0))] = 1

            except Exception as e:
                print(e)
                print(sys.exc_info()[0])

        # reconvert to sitk and restore information
        new_mask = sitk.GetImageFromArray(new_mask)
        new_mask.SetOrigin(origin)
        new_mask.SetDirection(direction)
        new_mask.SetSpacing(spacing)

        return new_mask


class Roi2MaskProbs(object):
    """
    Apply threshold-based method to calculate the non-binary (probs) segmentation from the ROI
    """

    def __init__(self, keys=('pet_img', 'mask_img'),mode = ['probs'], method=['otsu'],
                 new_key_name='mask_img'):
        """
        :param keys: (pet_img, roi_img) : (3D SimpleITK img, 4D SimpleITK img)
        :param method: method to use for calculate the threshold
                Must be one of 'absolute', 'relative', 'otsu', 'adaptative'
        """
        self.keys = (keys,) if isinstance(keys, str) else keys

        self.keys = keys
        self.mode = mode 
        self.method = method
        self.method = [self.method.lower()] if isinstance(self.method, str) else [el.lower() for el in self.method]
        self.new_key_name = new_key_name

        tvals_probs = dict(absolute=dict(lower=2.0, upper=4.0, mu=2.5, std=0.5),
                            relative=dict(lower=0.33, upper=0.60, mu=0.42, std=0.06))
        self.tvals_probs = tvals_probs

    def __call__(self, img_dict):
        pet_key = self.keys[0]
        roi_key = self.keys[1]

        img_dict[self.new_key_name] = self.roi2mask(img_dict[roi_key], img_dict[pet_key])
        return img_dict

    def relative_seg(self, roi):
        """Return a truncated mask array 
        """
        # lower, upper = 0.33, 0.60
        # mu, std = 0.42, 0.06

        lower, upper = self.tvals_probs['relative']['lower'], self.tvals_probs['relative']['upper']
        mu, std = self.tvals_probs['relative']['mu'], self.tvals_probs['relative']['std']

        a, b = (lower - mu) / std, (upper - mu) / std

        return truncnorm.cdf(roi / np.max(roi), a, b, loc=mu, scale=std)
        # return uniform.cdf(roi / np.max(roi), loc=lower, scale=upper - lower)

    def absolute_seg(self, roi):
        """Return a truncated mask array 
        """
        # lower, upper = 2.0, 4.0
        # mu, std = 2.5, 0.5

        lower, upper = self.tvals_probs['absolute']['lower'], self.tvals_probs['absolute']['upper']
        mu, std = self.tvals_probs['absolute']['mu'], self.tvals_probs['absolute']['std']

        a, b = (lower - mu) / std, (upper - mu) / std

        return truncnorm.cdf(roi, a, b, loc=mu, scale=std)
        # return uniform.cdf(roi, loc=lower, scale=upper - lower)

    def otsu_seg(self, roi):
        t = filters.threshold_otsu(roi)
        return np.where(roi > t, 1.0, 0.0)

    def compute_probs(self, roi, method):

        if method == 'absolute':
            return self.absolute_seg(roi)
        elif method == 'relative':
            return self.relative_seg(roi)
        elif method == 'otsu' or 'otsu_abs':
            return self.otsu_seg(roi)
        else:
            raise ValueError("method '{}' not supported. please use one of {}".format(method, "|".join(
                ['absolute', 'relative', 'otsu', 'otsu_abs'])))

    def roi2mask(self, mask_img, merged_img):
        """
        Generate the mask from the ROI of the pet scan
        Args:
            :param mask_img: sitk image, raw mask (i.e ROI)
            :param pet_img: sitk image, the corresponding pet scan

        :return: sitk image, the ground truth segmentation
        """
        # transform to numpy
        mask_array = sitk.GetArrayFromImage(mask_img)
        pet_array = sitk.GetArrayFromImage(merged_img)

        # get 3D meta information
        if len(mask_array.shape) == 3:
            mask_array = np.expand_dims(mask_array, axis=0)

            origin = mask_img.GetOrigin()
            spacing = mask_img.GetSpacing()
            direction = tuple(mask_img.GetDirection())
            # size = mask_img.GetSize()
        else:
            # convert false-4d meta information to 3d information
            origin = mask_img.GetOrigin()[:-1]
            spacing = mask_img.GetSpacing()[:-1]
            direction = tuple(el for i, el in enumerate(mask_img.GetDirection()[:12]) if not (i + 1) % 4 == 0)
            # size = mask_img.GetSize()[:-1]


        #len(mask_array.shape) = 4 ! 

        if self.mode == 'probs' and self.method == 'otsu_abs' : 
            new_masks = []
            for method in ['otsu', 'absolute']:
                new_mask = np.zeros(mask_array.shape[1:], dtype=np.float64)

                for num_slice in range(mask_array.shape[0]):
                    mask_slice = mask_array[num_slice]  # R.O.I
                    roi = pet_array[mask_slice > 0]
                    if len(roi) == 0:
                        continue
                    try:
                        # apply threshold
                        new_mask[np.where(mask_slice > 0)] = np.maximum(self.compute_probs(roi, method),
                                                                        new_mask[np.where(mask_slice > 0)])
                    except Exception as e:
                        print(e)
                        print(sys.exc_info()[0])
                new_masks.append(new_mask)

            new_mask = new_masks[0] if len(new_masks) == 1 else np.prod(np.array(new_masks), axis=0)

            # reconvert to sitk and restore 3D meta-information
            new_mask = sitk.GetImageFromArray(new_mask)
            new_mask.SetOrigin(origin)
            new_mask.SetDirection(direction)
            new_mask.SetSpacing(spacing)

            return new_mask

        else : 
            new_masks = []
            for method in self.method:
                new_mask = np.zeros(mask_array.shape[1:], dtype=np.float64) #len(new_mask.shape) = 3

                for num_slice in range(mask_array.shape[0]):
                    mask_slice = mask_array[num_slice]  # R.O.I #matrix 3D
                    roi = pet_array[mask_slice > 0] #vecteur 
                    if len(roi) == 0:
                        continue
                    try:
                        # apply threshold
                        new_mask[np.where(mask_slice > 0)] = np.maximum(self.compute_probs(roi, method),
                                                                        new_mask[np.where(mask_slice > 0)])
                    except Exception as e:
                        print(e)
                        print(sys.exc_info()[0])
                new_masks.append(new_mask)

            new_mask = new_masks[0] if len(new_masks) == 1 else np.mean(np.array(new_masks), axis=0)
            # np.average(imgs, axis=0, weights=self.weights)
            #new mask = mean de toutes les ROI selon la méthode 


            # reconvert to sitk and restore 3D meta-information
            new_mask = sitk.GetImageFromArray(new_mask)
            new_mask.SetOrigin(origin)
            new_mask.SetDirection(direction)
            new_mask.SetSpacing(spacing)

            return new_mask


class AverageImage(object):
    """A class to applied average on several ndarray


    """

    def __init__(self, keys, new_key_name, weights=None):
        self.keys = (keys,) if isinstance(keys, str) else keys
        self.weights = weights
        self.new_key_name = new_key_name

    def __call__(self, img_dict):
        imgs = np.array([img_dict.pop(key) for key in self.keys])
        if self.weights is None:
            img_dict[self.new_key_name] = np.mean(imgs, axis=0)
        else:
            img_dict[self.new_key_name] = np.average(imgs, axis=0, weights=self.weights)

        return img_dict




class RandAffine(object):
    """
    Data Augmentation for 3D SimpleITK image.
    The same random deformation is applied for each key (CT, PET, MASK).
    """

    def __init__(self, keys=('pet_img', 'ct_img', 'mask_img'),
                 translation=10, scaling=0.1, rotation=(0.0, pi / 30, 0.0)):
        """
        :param keys: str or tuple(str)
        :param translation: float, int or 3D tuple(float, int)
        :param scaling: float, int or 3D tuple(float, int)
        :param rotation: float, int or 3D tuple(float, int)
        :param default_value: dict(float, int)
        :param interpolator: dict(sitk.interpolator mode).
        """

            # sitk.sitkLinear, sitk.sitkBSpline, sitk.sitkNearestNeighbor
        if keys==('pet_img', 'ct_img', 'mask_img'):    
            interpolator = {'pet_img': sitk.sitkBSpline,
                                'ct_img': sitk.sitkBSpline,
                                'mask_img': sitk.sitkNearestNeighbor}
            default_value = {'pet_img': 0.0, 'ct_img': -1000.0, 'mask_img': 0}
        elif keys==('pet_img', 'ct_img'):
            interpolator = {'pet_img': sitk.sitkBSpline,
                                'ct_img': sitk.sitkBSpline}
            default_value = {'pet_img': 0.0, 'ct_img': -1000.0}
        
        

        self.keys = (keys,) if isinstance(keys, str) else keys

        self.translation = translation if isinstance(translation, tuple) else (translation, translation, translation)
        self.scaling = scaling if isinstance(scaling, tuple) else (scaling, scaling, scaling)
        self.rotation = rotation if isinstance(rotation, tuple) else (rotation, rotation, rotation)

        self.default_value = default_value
        self.interpolator = interpolator

    def __call__(self, img_dict):

        # generate deformation params
        def_ratios = self.generate_random_deformation_ratios()

        # apply the same deformation to every image
        for key in self.keys:
            img_dict[key] = self.AffineTransformation(image=img_dict.pop(key),
                                                      interpolator=self.interpolator[key],
                                                      deformations=def_ratios,
                                                      default_value=self.default_value[key])

        return img_dict

    @staticmethod
    def generate_random_bool(p):
        """
        :param p : float between 0-1, probability
        :return: True if a probability of p
        """
        return random.random() < p

    def generate_random_deformation_ratios(self):
        """
        :return: dict with random deformation
        """

        deformation = dict()
        if self.generate_random_bool(0.8):
            deformation['translation'] = (random.uniform(-1.0 * self.translation[0], self.translation[0]),
                                          random.uniform(-1.0 * self.translation[1], self.translation[1]),
                                          random.uniform(-1.0 * self.translation[2], self.translation[2]))
        else:
            deformation['translation'] = (0, 0, 0)

        if self.generate_random_bool(0.8):
            deformation['scaling'] = (random.uniform(1.0 - self.scaling[0], 1.0 + self.scaling[0]),
                                      random.uniform(1.0 - self.scaling[1], 1.0 + self.scaling[1]),
                                      random.uniform(1.0 - self.scaling[2], 1.0 + self.scaling[2]))
        else:
            deformation['scaling'] = (1.0, 1.0, 1.0)

        if self.generate_random_bool(0.8):
            deformation['rotation'] = (random.uniform(-1.0 * self.rotation[0], self.rotation[0]),
                                       random.uniform(-1.0 * self.rotation[1], self.rotation[1]),
                                       random.uniform(-1.0 * self.rotation[2], self.rotation[2]))
        else:
            deformation['rotation'] = (0.0, 0.0, 0.0)

        return deformation

    @staticmethod
    def AffineTransformation(image, interpolator, deformations, default_value):
        """
        Apply deformation to the input image
        :parameter
            :param image: Simple ITK image
            :param interpolator: method of interpolator, for ex : sitk.sitkBSpline
            :param deformations: dict of deformation to apply
            :param default_value: default value to fill the image
        :return: deformed image
        """

        center = tuple(
            np.asarray(image.GetOrigin()) + 0.5 * np.asarray(image.GetSize()) * np.asarray(image.GetSpacing()))

        transformation = sitk.AffineTransform(3)
        transformation.SetCenter(center)
        transformation.Scale(deformations['scaling'])
        transformation.Rotate(axis1=1, axis2=2, angle=deformations['rotation'][0])
        transformation.Rotate(axis1=0, axis2=2, angle=deformations['rotation'][1])
        transformation.Rotate(axis1=0, axis2=1, angle=deformations['rotation'][2])
        transformation.Translate(deformations['translation'])
        reference_image = image

        return sitk.Resample(image, reference_image, transformation, interpolator, default_value)



class Sitk2Numpy(object):
    """
    Convert SimpleITK image into Numpy ndarray
    """
    def __init__(self, keys=('pet_img', 'ct_img', 'mask_img')):
        self.keys = (keys,) if isinstance(keys, str) else keys

    def __call__(self, img_dict):
        for key in self.keys:
            img_dict[key] = sitk.GetArrayFromImage(img_dict.pop(key))
            # img = sitk.GetArrayFromImage(img_dict[key])
            # img = np.transpose(img, (2, 1, 0))  # (z, y, x) to (x, y, z)
            # img_dict[key] = img

        return img_dict


class ScaleIntensityRanged(object):
    """
    #Linearly Scale value between [a_min, a_max] to [b_min, b_max]. 
    Scale intensity of CT and PET from [a_min, a_max] to [b_min, b_max].
    """

    def __init__(self, keys, a_min, a_max, b_min, b_max, clip=False):
        self.keys = (keys,) if isinstance(keys, str) else keys

        self.a_min = a_min
        self.a_max = a_max
        self.b_min = b_min
        self.b_max = b_max
        self.clip = clip

        assert a_min < a_max
        assert b_min < b_max

    def __call__(self, img_dict):

        for key in self.keys:
            img = img_dict.pop(key)

            img = (img - self.a_min) / (self.a_max - self.a_min)
            img = img * (self.b_max - self.b_min) + self.b_min

            if self.clip:
                img = np.clip(img, self.b_min, self.b_max)

            img_dict[key] = img

        return img_dict


class ConcatModality(object):
    """
    expects data of shape (spatial_dim1, spatial_dim2, ..., spatial_dim3)
    Concatenate PET/CT array, returns a 4D array of 3D arrays.
    """

    def __init__(self, keys=('pet_img', 'ct_img'), channel_first=True, new_key='image'):
        self.keys = (keys,) if isinstance(keys, str) else keys
        self.channel_first = channel_first
        self.new_key = new_key

    def __call__(self, img_dict):
        idx_channel = 0 if self.channel_first else -1
        imgs = [img_dict.pop(key) for key in self.keys]
        img_dict[self.new_key] = np.stack(imgs, axis=idx_channel)

        return img_dict


class AddChannel(object):
    """
    expects data of shape (spatial_dim1, spatial_dim2, ..., spatial_dim3)
    Add a dimension to one ndarray : return a 4D array of one 3D array
    """

    def __init__(self, keys, channel_first=False):
        self.keys = (keys,) if isinstance(keys, str) else keys
        self.channel_first = channel_first

    def __call__(self, img_dict):
        axis = 0 if self.channel_first else -1
        for key in self.keys:
            img_dict[key] = np.expand_dims(img_dict[key], axis=axis)

        return img_dict


class RenameDict(object):
    """A class to rename key's dict
    """

    def __init__(self, keys, keys2):
        """
        :param keys: str or tuple(str), key to rename
        :param keys2: str or tuple(str), new name of keys
        """
        self.keys = (keys,) if isinstance(keys, str) else keys
        self.keys2 = (keys2,) if isinstance(keys2, str) else keys2

    def __call__(self, img_dict):
        for key1, key2 in zip(self.keys, self.keys2):
            img_dict[key2] = img_dict.pop(key1)

        return img_dict



class PostCNNResampler(object):
    """ A class to generate sitk Image of predicted mask

    """

    def __init__(self, threshold_prob=0.5):
        """
        :param threshold_prob: set to None to yield probs mask
        """
        self.threshold_prob = threshold_prob
        self.mode = 'Linear'
        assert self.mode in ['Linear', 'NearestNeighbor']
        if self.mode == 'NearestNeighbor':
            assert self.threshold_prob is not None

    def __call__(self, img_dict):
        mask_img = sitk.GetImageFromArray(img_dict['mask_pred'])
        if self.mode == 'NearestNeighbor':
            mask_img = np.where(mask_img > self.threshold_prob, 1, 0).astype(int)
        mask_img.SetOrigin(img_dict['meta_info']['new_origin'])
        mask_img.SetDirection(img_dict['meta_info']['new_direction'])
        mask_img.SetSpacing(img_dict['meta_info']['new_spacing'])

        # resample to orginal shape, spacing, direction and origin
        transformation = sitk.ResampleImageFilter()
        transformation.SetOutputDirection(img_dict['meta_info']['original_direction'])
        transformation.SetOutputOrigin(img_dict['meta_info']['original_origin'])
        transformation.SetOutputSpacing(img_dict['meta_info']['original_spacing'])
        transformation.SetSize(img_dict['meta_info']['original_size'])

        transformation.SetDefaultPixelValue(0.0)
        if self.mode == 'NearestNeighbor':
            transformation.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            transformation.SetInterpolator(sitk.sitkLinear)
        mask_img_final = transformation.Execute(mask_img)

        if self.threshold_prob is not None and self.mode != 'NearestNeighbor':
            mask_img_final = sitk.BinaryThreshold(mask_img_final,
                                                  lowerThreshold=0.0, upperThreshold=self.threshold_prob,
                                                  insideValue=0, outsideValue=1)
        return mask_img_final



