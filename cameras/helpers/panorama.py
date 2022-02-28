"""OpenCV panorama composite image stitching class"""

import cv2 as cv
import numpy as np
import os
import math

# Warp Options
VK_PANORAMA_WARP_SPHERICAL = "spherical"
VK_PANORAMA_WARP_PLANE = "plane"
VK_PANORAMA_WARP_AFFINE = "affine"
VK_PANORAMA_WARP_CYLINDIRICAL = "cylindrical"
VK_PANORAMA_WARP_FISHEYE = "fisheye"
VK_PANORAMA_WARP_STEREOGRAPHIC = "stereographic"
VK_PANORAMA_WARP_COMPRESSEDPLANE_A2B1 = "compressedPlaneA2B1",
VK_PANORAMA_WARP_COMPRESSEDPLANE_A15B1 = "compressedPlaneA1.5B1"
VK_PANORAMA_WARP_COMPRESSEDPLANE_PORTRAIT_A2B1 = "compressedPlanePortraitA2B1"
VK_PANORAMA_WARP_COMPRESSEDPLANE_PORTRAIT_A15B1 = "compressedPlanePortraitA1.5B1"
VK_PANORAMA_WARP_PANINI_A2B1 = "paniniA2B1"
VK_PANORAMA_WARP_PANINI_A15B1 = "paniniA1.5B1"
VK_PANORAMA_WARP_PANINI_PORTRAIT_A2B1 = "paniniPortraitA2B1"
VK_PANORAMA_WARP_PANINI_PORTRAIT_A15B1 = "paniniPortraitA1.5B1"
VK_PANORAMA_WARP_MERCATOR = "mercator"
VK_PANORAMA_WARP_TRANSVERSE_MERCATOR = "transverseMercator"

# Blend Options
VK_PANORAMA_BLEND_MULTIBAND = "spherical"
VK_PANORAMA_BLEND_FEATHER = "feather"

# Blend Feature Match Algorithm
VK_PANORAMA_FEATURE_BRISK = "BRISK"
VK_PANORAMA_FEATURE_AKAZE = "AKAZE"


def get_feature_matcher():
    """Create feature matching object.

    Returns:
        (FeaturesMatcher): Features matcher which finds best matches for each feature.
    """
    try_cuda = True
    match_conf = 0.65
    matcher = cv.detail.BestOf2NearestMatcher_create(try_cuda, match_conf)
    return matcher


def get_exposure_compensator():
    """Compensate exposure in the specified image.

    Returns:
        (ExposureCompensator): Calibrated exposure compensator.
    """
    expos_comp_type = cv.detail.ExposureCompensator_GAIN_BLOCKS
    expos_comp_nr_feeds = 1
    expos_comp_block_size = 32

    if expos_comp_type == cv.detail.ExposureCompensator_CHANNELS:
        compensator = cv.detail_ChannelsCompensator(expos_comp_nr_feeds)

    elif expos_comp_type == cv.detail.ExposureCompensator_CHANNELS_BLOCKS:
        compensator = cv.detail_BlocksChannelsCompensator(
            expos_comp_block_size, expos_comp_block_size,
            expos_comp_nr_feeds
        )

    else:
        compensator = cv.detail.ExposureCompensator_createDefault(expos_comp_type)
    return compensator


class VKPanoramaController:
    """Constructor for panoramic composite image stitching controller class.

    Args:
        params (dict): Default parameters override.
    """
    def __init__(self, params):

        assert params is not None, "Invalid stitching params..."
        assert params.__class__.__name__ == "dict", "Invalid stitching params..."

        # Initialise global operators.
        self.compose_work_aspect = 0
        self.warped_image_scale = 1
        self.compensator = get_exposure_compensator()
        self.masks_warped = []
        self.corners = []
        self.sizes = []

        # Scaling factor used to downscale image resolution for image keypoint matching.
        if "work_megapix" in params:
            self.work_megapix = params["work_megapix"]
        else:
            self.work_megapix = 0.6

        # Feature matching algorithm
        if "feature_match_algorithm" in params:
            self.feature_match_algorithm = params["feature_match_algorithm"]
        else:
            self.feature_match_algorithm = VK_PANORAMA_FEATURE_BRISK

        # Warping mode.
        if "warp_type" in params:
            self.warp_type = params["warp_type"]
        else:
            self.warp_type = VK_PANORAMA_WARP_SPHERICAL

        # Wave correct mode
        if "wave_correct" in params:
            self.wave_correct = params["wave_correct"]
        else:
            self.wave_correct = 'horiz'

        # Wave correct mode
        if "blend_type" in params:
            self.blend_type = params["blend_type"]
        else:
            self.blend_type = VK_PANORAMA_BLEND_MULTIBAND

        # Wave correct mode
        if "blend_strength" in params:
            self.blend_strength = params["blend_strength"]
        else:
            self.blend_strength = 5

    def __str__(self):
        """Overriding str

        Returns:
            (str): A string summary of the object
        """
        return "\nPanoramic Composite Image Parameters:" \
               "\n\twork_megapix:   {0}" \
               "\n\twarp_type:      {1}" \
               "\n\tblend_type:     {2}" \
               "\n\tblend_strength: {3}" \
               "\n\twave_correct:   {4}".format(self.work_megapix,
                                                self.warp_type,
                                                self.blend_type,
                                                self.blend_strength,
                                                self.wave_correct)

    def compute_transforms(self, input_images, input_names):
        """Compute initial composite matrices and properties.
        This should only be run once for each camera input set at initialisation.
        Subsequent frame-wise panoramas should re-use these matrices.

        Args:
            input_images (list): A list of image arrays.
            input_names (list): Filenames for each image.

        Returns:
            dst (array): Composite image panorama.
            camera_models (list): List of camera objects with extrinsics and rotation matrices.
        """

        assert len(input_images) == len(input_names), "The input images should be of the same number as the input names..."
        camera_models = []

        seam_megapix = 0.1
        compose_megapix = -1
        ba_refine_mask = 'xxxxx'

        finder = None

        if self.feature_match_algorithm == "BRISK":
            try:
                finder = cv.BRISK_create()
                print("Created BRISK Feature Matching")
            except AttributeError:
                print("BRISK not available")

        else:
            try:
                finder = cv.AKAZE_create()
                print("Created AKAZE Feature Matching")
            except AttributeError:
                print("AKAZE not available")

        assert finder is not None, "No feature matching found.."

        seam_work_aspect = 1
        full_img_sizes = []
        features = []
        images = []
        is_work_scale_set = False
        is_seam_scale_set = False
        is_compose_scale_set = False
        work_scale = 1
        seam_scale = 1

        '''
        RESIZE IMAGES BY SCALING FACTOR
        - Pattern matching is done at low resolution for performance.
        '''
        for full_img in input_images:
            full_img_sizes.append((full_img.shape[1], full_img.shape[0]))
            if self.work_megapix < 0:
                img = full_img
                work_scale = 1
                is_work_scale_set = True
            else:
                if is_work_scale_set is False:
                    work_scale = min(1.0, np.sqrt(self.work_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
                    is_work_scale_set = True
                img = cv.resize(src=full_img, dsize=None, fx=work_scale, fy=work_scale,
                                interpolation=cv.INTER_LINEAR_EXACT)
            if is_seam_scale_set is False:
                seam_scale = min(1.0, np.sqrt(seam_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
                seam_work_aspect = seam_scale / work_scale
                is_seam_scale_set = True
            img_feat = cv.detail.computeImageFeatures2(finder, img)
            features.append(img_feat)
            img = cv.resize(src=full_img, dsize=None, fx=seam_scale, fy=seam_scale, interpolation=cv.INTER_LINEAR_EXACT)
            images.append(img)

        '''
        FIND FEATURE MATCHES AT LOW RESOLUTION
        '''
        matcher = get_feature_matcher()
        p = matcher.apply2(features)
        matcher.collectGarbage()

        indices = cv.detail.leaveBiggestComponent(features, p, 0.3)
        img_subset = []
        img_names_subset = []
        full_img_sizes_subset = []
        for i in range(len(indices)):
            img_names_subset.append(input_names[indices[i, 0]])
            img_subset.append(images[indices[i, 0]])
            full_img_sizes_subset.append(full_img_sizes[indices[i, 0]])
        images = img_subset
        input_names = img_names_subset
        full_img_sizes = full_img_sizes_subset
        num_images = len(input_names)
        if num_images < 2:
            print("Need more images")
            return None, None

        '''
        Initialise Homography-based rotation estimator
        '''
        estimator = cv.detail_HomographyBasedEstimator()
        b, cameras = estimator.apply(features, p, None)
        if not b:
            return None, None

        for idx, cam in enumerate(cameras):
            cam.R = cam.R.astype(np.float32)

        '''
        Refine rotation estimators
        '''
        adjuster = cv.detail_BundleAdjusterRay()
        adjuster.setConfThresh(1)
        refine_mask = np.zeros((3, 3), np.uint8)
        if ba_refine_mask[0] == 'x':
            refine_mask[0, 0] = 1
        if ba_refine_mask[1] == 'x':
            refine_mask[0, 1] = 1
        if ba_refine_mask[2] == 'x':
            refine_mask[0, 2] = 1
        if ba_refine_mask[3] == 'x':
            refine_mask[1, 1] = 1
        if ba_refine_mask[4] == 'x':
            refine_mask[1, 2] = 1
        adjuster.setRefinementMask(refine_mask)
        b, cameras = adjuster.apply(features, p, cameras)
        if not b:
            print("Camera parameters adjusting failed.")
            return None, None

        '''
        Refine rotation estimators
        '''
        focals = []
        for cam in cameras:
            focals.append(cam.focal)
        sorted(focals)
        if len(focals) % 2 == 1:
            self.warped_image_scale = focals[len(focals) // 2]
        else:
            self.warped_image_scale = (focals[len(focals) // 2] + focals[len(focals) // 2 - 1]) / 2

        rmats = []
        for cam in cameras:
            rmats.append(np.copy(cam.R))

        if self.wave_correct == "vert":
            k = cv.detail.WAVE_CORRECT_VERT
        else:
            k = cv.detail.WAVE_CORRECT_HORIZ

        rmats = cv.detail.waveCorrect(rmats, k)
        for idx, cam in enumerate(cameras):
            cam.R = rmats[idx]

        images_warped = []
        masks = []
        for i in range(0, num_images):
            um = cv.UMat(255 * np.ones((images[i].shape[0], images[i].shape[1]), np.uint8))
            masks.append(um)

        warper = cv.PyRotationWarper(self.warp_type, self.warped_image_scale * seam_work_aspect)  # warper could be nullptr?
        for idx in range(0, num_images):
            K = cameras[idx].K().astype(np.float32)
            swa = seam_work_aspect
            K[0, 0] *= swa
            K[0, 2] *= swa
            K[1, 1] *= swa
            K[1, 2] *= swa

            # Rotate low-resolution images.
            corner, image_wp = warper.warp(images[idx], K, cameras[idx].R, cv.INTER_LINEAR, cv.BORDER_REFLECT)

            self.corners.append(corner)
            self.sizes.append((image_wp.shape[1], image_wp.shape[0]))

            images_warped.append(image_wp)

            p, mask_wp = warper.warp(masks[idx], K, cameras[idx].R, cv.INTER_NEAREST, cv.BORDER_CONSTANT)
            self.masks_warped.append(mask_wp.get())

        images_warped_f = []
        for img in images_warped:
            imgf = img.astype(np.float32)
            images_warped_f.append(imgf)

        self.compensator.feed(corners=self.corners, images=images_warped, masks=self.masks_warped)

        seam_finder = cv.detail_GraphCutSeamFinder('COST_COLOR')
        seam_finder.find(images_warped_f, self.corners, self.masks_warped)
        compose_scale = 1
        self.corners = []
        self.sizes = []
        blender = None

        '''
        Warp full size images with camera matrices
        '''
        # https://github.com/opencv/opencv/blob/master/samples/cpp/stitching_detailed.cpp#L725 ?
        for idx, full_img in enumerate(input_images):
            if not is_compose_scale_set:
                if compose_megapix > 0:
                    compose_scale = min(1.0, np.sqrt(compose_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
                is_compose_scale_set = True
                self.compose_work_aspect = compose_scale / work_scale

                print("self.compose_work_aspect = compose_scale / work_scale")
                print(self.compose_work_aspect, compose_scale, work_scale)

                self.warped_image_scale *= self.compose_work_aspect

                print("self.warped_image_scale = {0}".format(self.warped_image_scale))

                # Make the warper first time around.
                warper = cv.PyRotationWarper(self.warp_type, self.warped_image_scale)

                # Update all cameras with self.compose_work_aspect
                for i in range(0, len(input_images)):
                    cameras[i].focal *= self.compose_work_aspect
                    cameras[i].ppx *= self.compose_work_aspect
                    cameras[i].ppy *= self.compose_work_aspect
                    sz = (full_img_sizes[i][0] * compose_scale, full_img_sizes[i][1] * compose_scale)
                    K = cameras[i].K().astype(np.float32)
                    roi = warper.warpRoi(sz, K, cameras[i].R)
                    self.corners.append(roi[0:2])
                    self.sizes.append(roi[2:4])

            if abs(compose_scale - 1) > 1e-1:
                img = cv.resize(src=full_img, dsize=None, fx=compose_scale, fy=compose_scale,
                                interpolation=cv.INTER_LINEAR_EXACT)
            else:
                img = full_img
            _img_size = (img.shape[1], img.shape[0])
            K = cameras[idx].K().astype(np.float32)

            corner, image_warped = warper.warp(img, K, cameras[idx].R, cv.INTER_LINEAR, cv.BORDER_REFLECT)

            short_name = os.path.splitext(os.path.splitext(os.path.basename(input_names[idx]))[0])[0]

            camera_models.append({"extrinsics": K,
                                  "rotation": cameras[idx].R,
                                  "corner": corner,
                                  "name": input_names[idx],
                                  "short_name": short_name})

            mask = 255 * np.ones((img.shape[0], img.shape[1]), np.uint8)
            p, mask_warped = warper.warp(mask, K, cameras[idx].R, cv.INTER_NEAREST, cv.BORDER_CONSTANT)
            self.compensator.apply(idx, self.corners[idx], image_warped, mask_warped)
            image_warped_s = image_warped.astype(np.int16)
            dilated_mask = cv.dilate(self.masks_warped[idx], None)
            seam_mask = cv.resize(dilated_mask, (mask_warped.shape[1], mask_warped.shape[0]), 0, 0, cv.INTER_LINEAR_EXACT)
            mask_warped = cv.bitwise_and(seam_mask, mask_warped)

            if blender is None:
                blender = cv.detail.Blender_createDefault(cv.detail.Blender_NO)
                dst_sz = cv.detail.resultRoi(corners=self.corners, sizes=self.sizes)

                blend_width = np.sqrt(dst_sz[2] * dst_sz[3]) * self.blend_strength / 100
                if blend_width < 1:
                    blender = cv.detail.Blender_createDefault(cv.detail.Blender_NO)
                elif self.blend_type == "multiband":
                    blender = cv.detail_MultiBandBlender()
                    blender.setNumBands((np.log(blend_width) / np.log(2.) - 1.).astype(np.int))
                elif self.blend_type == "feather":
                    blender = cv.detail_FeatherBlender()
                    blender.setSharpness(1. / blend_width)
                blender.prepare(dst_sz)

            blender.feed(cv.UMat(image_warped_s), mask_warped, self.corners[idx])

        result = None
        result_mask = None
        result, result_mask = blender.blend(result, result_mask)
        dst = cv.normalize(src=result, dst=None, alpha=255., norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

        # Markers for sanity
        # NB - camera offsets are relative to left-most camera.
        _reference_frame = 0
        _left_max = math.inf
        for idx, m in enumerate(camera_models):
            if _left_max > m["corner"][0]:
                _reference_frame = idx
                _left_max = min(_left_max, m["corner"][0])

        print("Reference Frame = {0}".format(_reference_frame))

        for idx, img in enumerate(input_images):
            print("\nModel for {0}:".format(input_names[idx]))
            for k in camera_models[idx]:
                print("{0}:\n{1}".format(k, camera_models[idx][k]))
            dx = camera_models[_reference_frame]["corner"][0]
            dy = camera_models[_reference_frame]["corner"][1]
            K = camera_models[idx]["extrinsics"]

            for x in (0, img.shape[1]):
                for y in (0, img.shape[0]):
                    pt = warper.warpPoint((x, y), K, cameras[idx].R)
                    pt = (int(pt[0]), int(pt[1]))
                    pt = (int(pt[0]) - dx, int(pt[1]) - dy)
                    cv.drawMarker(dst, pt, (255, 255, 255), cv.MARKER_CROSS, thickness=4)

        return dst, camera_models

    def stitch(self, panorama_projection_models, input_images, camera_models=None, annotations=None):
        """Compute frame-wise panoramas.

        Args:
            panorama_projection_models (list): List of panorama projection objects with extrinsics and rotation matrices.
            These are distinct from the core camera calibration models.
            input_images (list): A list of image arrays.
            camera_models (list): Optional - A list of core camera objects.
            Required for annotating from world space to camera space, then to panorama space.
            annotations (list): Future property to add image annotations.

        Returns:
            dst (array): Composite image panorama.
        """

        assert len(panorama_projection_models) > 0, "Camera models are required..."

        blender = None

        '''
        Warp full size images with camera pre-computed matrices
        '''
        # Make the warper first time around.
        warper = cv.PyRotationWarper(self.warp_type, self.warped_image_scale)

        for idx, img in enumerate(input_images):
            _img_size = (img.shape[1], img.shape[0])
            K = panorama_projection_models[idx]["extrinsics"].astype(np.float32)
            R = panorama_projection_models[idx]["rotation"].astype(np.float32)
            corner, image_warped = warper.warp(img, K, R, cv.INTER_LINEAR, cv.BORDER_REFLECT)

            mask = 255 * np.ones((img.shape[0], img.shape[1]), np.uint8)
            p, mask_warped = warper.warp(mask, K, R, cv.INTER_NEAREST, cv.BORDER_CONSTANT)
            self.compensator.apply(idx, corner, image_warped, mask_warped)
            image_warped_s = image_warped.astype(np.int16)
            dilated_mask = cv.dilate(self.masks_warped[idx], None)
            seam_mask = cv.resize(dilated_mask, (mask_warped.shape[1], mask_warped.shape[0]), 0, 0,
                                  cv.INTER_LINEAR_EXACT)
            mask_warped = cv.bitwise_and(seam_mask, mask_warped)

            if blender is None:
                blender = cv.detail.Blender_createDefault(cv.detail.Blender_NO)
                dst_sz = cv.detail.resultRoi(corners=self.corners, sizes=self.sizes)

                blend_width = np.sqrt(dst_sz[2] * dst_sz[3]) * self.blend_strength / 100
                if blend_width < 1:
                    blender = cv.detail.Blender_createDefault(cv.detail.Blender_NO)
                elif self.blend_type == "multiband":
                    blender = cv.detail_MultiBandBlender()
                    blender.setNumBands((np.log(blend_width) / np.log(2.) - 1.).astype(np.int))
                elif self.blend_type == "feather":
                    blender = cv.detail_FeatherBlender()
                    blender.setSharpness(1. / blend_width)
                blender.prepare(dst_sz)

            blender.feed(cv.UMat(image_warped_s), mask_warped, self.corners[idx])

        result = None
        result_mask = None
        result, result_mask = blender.blend(result, result_mask)
        dst = cv.normalize(src=result, dst=None, alpha=255., norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

        if camera_models is not None:
            if annotations:

                for _player_id, annotation in enumerate(annotations):

                    panoramic_image_point = self.panoramic_point_for_world_point(world_point=annotation["unified_world_foot"],
                                                                                 panorama_projection_models=panorama_projection_models,
                                                                                 camera_models=camera_models)

                    cv.circle(dst, panoramic_image_point, radius=5, color=(255, 0, 0), thickness=4)
                    cv.putText(dst, str(_player_id), panoramic_image_point, cv.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 0), 2, cv.LINE_AA)

        return dst

    def panoramic_point_for_world_point(self, world_point, panorama_projection_models, camera_models):

        warper = cv.PyRotationWarper(self.warp_type, self.warped_image_scale)

        # Draw annotations
        # TODO - auto select which camera to use as the offset??
        dx = panorama_projection_models[0]["corner"][0]
        dy = panorama_projection_models[0]["corner"][1]

        _scale = camera_models[0].surface_model.surface_properties["model_scale"]
        _offset = camera_models[0].surface_model.surface_properties["model_offset_x"] * _scale

        print("\n****> ANNOTATING:", world_point)
        x, y = world_point
        x = x * _scale
        y = y * _scale

        # So now we have the image point from the ground point for this camera model.
        # We need to project that image point into the panorama view composite.

        model_point = np.array([[[x, y, 0]]], dtype='float32')

        for idx, camera_model in enumerate(camera_models):

            # Project model point to the local camera point.
            camera_wise_image_point = camera_model.surface_model.projected_image_point_for_3d_world_point(world_point=model_point, camera_model=camera_model)

            x, y = camera_wise_image_point[0][0]
            if 0 < x < camera_model.width():
                if 0 < y < camera_model.height():
                    # print("Camera {0} is IN bounds".format(idx))
                    # What are the bounds on the image?
                    panorama_projection = panorama_projection_models[idx]

                    # Project local camera point to panoramic image point.
                    pt = warper.warpPoint((x, y), panorama_projection["extrinsics"], panorama_projection["rotation"])
                    pt = (int(pt[0]), int(pt[1]))
                    pt = (int(pt[0]) - dx, int(pt[1]) - dy)
                    print("Image pt to panorama pt:", pt)
                    return pt

        return None, None
