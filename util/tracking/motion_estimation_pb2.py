# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/util/tracking/motion_estimation.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n/mediapipe/util/tracking/motion_estimation.proto\x12\tmediapipe\"\xb7\x35\n\x17MotionEstimationOptions\x12\'\n\x19\x65stimate_translation_irls\x18\x01 \x01(\x08:\x04true\x12w\n\x1clinear_similarity_estimation\x18\x03 \x01(\x0e\x32=.mediapipe.MotionEstimationOptions.LinearSimilarityEstimation:\x12\x45STIMATION_LS_IRLS\x12\x66\n\x11\x61\x66\x66ine_estimation\x18\x1e \x01(\x0e\x32\x33.mediapipe.MotionEstimationOptions.AffineEstimation:\x16\x45STIMATION_AFFINE_NONE\x12m\n\x15homography_estimation\x18\x05 \x01(\x0e\x32\x37.mediapipe.MotionEstimationOptions.HomographyEstimation:\x15\x45STIMATION_HOMOG_IRLS\x12\x33\n$homography_exact_denominator_scaling\x18\x35 \x01(\x08:\x05\x66\x61lse\x12-\n\x1fuse_exact_homography_estimation\x18\x36 \x01(\x08:\x04true\x12\x37\n)use_highest_accuracy_for_normal_equations\x18\x37 \x01(\x08:\x04true\x12-\n\"homography_perspective_regularizer\x18= \x01(\x02:\x01\x30\x12|\n\x19mix_homography_estimation\x18\x0c \x01(\x0e\x32>.mediapipe.MotionEstimationOptions.MixtureHomographyEstimation:\x19\x45STIMATION_HOMOG_MIX_NONE\x12\x18\n\x0cnum_mixtures\x18\r \x01(\x05:\x02\x31\x30\x12\x1e\n\x11mixture_row_sigma\x18\x0e \x01(\x02:\x03\x30.1\x12#\n\x13mixture_regularizer\x18\x0f \x01(\x02:\x06\x30.0001\x12%\n\x1amixture_regularizer_levels\x18* \x01(\x02:\x01\x33\x12%\n\x18mixture_regularizer_base\x18+ \x01(\x02:\x03\x32.2\x12$\n\x19mixture_rs_analysis_level\x18, \x01(\x05:\x01\x32\x12\x17\n\x0birls_rounds\x18\x11 \x01(\x05:\x02\x31\x30\x12\x1d\n\x10irls_prior_scale\x18\x32 \x01(\x02:\x03\x30.2\x12,\n\x1eirls_motion_magnitude_fraction\x18\x1f \x01(\x02:\x04\x30.08\x12(\n\x1birls_mixture_fraction_scale\x18\x44 \x01(\x02:\x03\x31.5\x12*\n\x1birls_weights_preinitialized\x18\' \x01(\x08:\x05\x66\x61lse\x12.\n\x1f\x66ilter_initialized_irls_weights\x18\x43 \x01(\x08:\x05\x66\x61lse\x12Y\n\x13irls_initialization\x18\x38 \x01(\x0b\x32<.mediapipe.MotionEstimationOptions.IrlsOutlierInitialization\x12,\n\x1d\x66\x65\x61ture_density_normalization\x18> \x01(\x08:\x05\x66\x61lse\x12\x1d\n\x11\x66\x65\x61ture_mask_size\x18? \x01(\x05:\x02\x31\x30\x12\x61\n\x1blong_feature_initialization\x18\x42 \x01(\x0b\x32<.mediapipe.MotionEstimationOptions.LongFeatureInitialization\x12M\n\x11irls_mask_options\x18\x39 \x01(\x0b\x32\x32.mediapipe.MotionEstimationOptions.IrlsMaskOptions\x12^\n\x16joint_track_estimation\x18; \x01(\x0b\x32>.mediapipe.MotionEstimationOptions.JointTrackEstimationOptions\x12\\\n\x19long_feature_bias_options\x18@ \x01(\x0b\x32\x39.mediapipe.MotionEstimationOptions.LongFeatureBiasOptions\x12\x64\n\x11\x65stimation_policy\x18: \x01(\x0e\x32\x33.mediapipe.MotionEstimationOptions.EstimationPolicy:\x14INDEPENDENT_PARALLEL\x12\x1e\n\x12\x63overage_grid_size\x18\x33 \x01(\x05:\x02\x31\x30\x12\x66\n\x12mixture_model_mode\x18\x17 \x01(\x0e\x32\x33.mediapipe.MotionEstimationOptions.MixtureModelMode:\x15SKEW_ROTATION_MIXTURE\x12\x35\n\'use_only_lin_sim_inliers_for_homography\x18\x06 \x01(\x08:\x04true\x12\'\n\x18lin_sim_inlier_threshold\x18\x14 \x01(\x02:\x05\x30.003\x12W\n\x19stable_translation_bounds\x18  \x01(\x0b\x32\x34.mediapipe.MotionEstimationOptions.TranslationBounds\x12U\n\x18stable_similarity_bounds\x18! \x01(\x0b\x32\x33.mediapipe.MotionEstimationOptions.SimilarityBounds\x12U\n\x18stable_homography_bounds\x18\x0b \x01(\x0b\x32\x33.mediapipe.MotionEstimationOptions.HomographyBounds\x12\x64\n stable_mixture_homography_bounds\x18\" \x01(\x0b\x32:.mediapipe.MotionEstimationOptions.MixtureHomographyBounds\x12$\n\x15strict_coverage_scale\x18) \x01(\x02:\x05\x31.333\x12)\n\x1blabel_empty_frames_as_valid\x18\x16 \x01(\x08:\x04true\x12\x1f\n\x11\x66\x65\x61ture_grid_size\x18\x18 \x01(\x02:\x04\x30.05\x12\x1b\n\rspatial_sigma\x18\x19 \x01(\x02:\x04\x30.01\x12\"\n\x16temporal_irls_diameter\x18\x1a \x01(\x05:\x02\x32\x30\x12\x19\n\x0etemporal_sigma\x18\x1b \x01(\x02:\x01\x35\x12\x19\n\rfeature_sigma\x18\x1c \x01(\x02:\x02\x33\x30\x12\x1c\n\rfilter_5_taps\x18\x1d \x01(\x08:\x05\x66\x61lse\x12(\n\x1a\x66rame_confidence_weighting\x18\x30 \x01(\x08:\x04true\x12\'\n\x1areset_confidence_threshold\x18\x31 \x01(\x02:\x03\x30.4\x12\x61\n\x12irls_weight_filter\x18# \x01(\x0e\x32\x33.mediapipe.MotionEstimationOptions.IRLSWeightFilter:\x10IRLS_FILTER_NONE\x12 \n\x11overlay_detection\x18$ \x01(\x08:\x05\x66\x61lse\x12&\n\x1boverlay_analysis_chunk_size\x18% \x01(\x05:\x01\x38\x12]\n\x19overlay_detection_options\x18& \x01(\x0b\x32:.mediapipe.MotionEstimationOptions.OverlayDetectionOptions\x12U\n\x15shot_boundary_options\x18< \x01(\x0b\x32\x36.mediapipe.MotionEstimationOptions.ShotBoundaryOptions\x12)\n\x1boutput_refined_irls_weights\x18( \x01(\x08:\x04true\x12\x94\x01\n%homography_irls_weight_initialization\x18- \x01(\x0e\x32\x45.mediapipe.MotionEstimationOptions.HomographyIrlsWeightInitialization:\x1eIRLS_WEIGHT_PERIMETER_GAUSSIAN\x12\x1e\n\x10irls_use_l0_norm\x18. \x01(\x08:\x04true\x12*\n\x1b\x64omain_limited_irls_scaling\x18\x41 \x01(\x08:\x05\x66\x61lse\x12\x32\n#deactivate_stable_motion_estimation\x18/ \x01(\x08:\x05\x66\x61lse\x12)\n\x1aproject_valid_motions_down\x18\x34 \x01(\x08:\x05\x66\x61lse\x12\x1f\n\x13\x65stimate_similarity\x18\x02 \x01(\x08\x42\x02\x18\x01\x1a\x61\n\x19IrlsOutlierInitialization\x12\x18\n\tactivated\x18\x01 \x01(\x08:\x05\x66\x61lse\x12\x13\n\x06rounds\x18\x02 \x01(\x05:\x03\x31\x30\x30\x12\x15\n\x06\x63utoff\x18\x03 \x01(\x02:\x05\x30.003\x1az\n\x19LongFeatureInitialization\x12\x18\n\tactivated\x18\x01 \x01(\x08:\x05\x66\x61lse\x12#\n\x15min_length_percentile\x18\x02 \x01(\x02:\x04\x30.95\x12\x1e\n\x13upweight_multiplier\x18\x03 \x01(\x02:\x01\x35\x1a\xd3\x01\n\x0fIrlsMaskOptions\x12\x12\n\x05\x64\x65\x63\x61y\x18\x02 \x01(\x02:\x03\x30.7\x12\x19\n\x0cinlier_score\x18\x03 \x01(\x02:\x03\x30.4\x12\x17\n\nbase_score\x18\x04 \x01(\x02:\x03\x30.2\x12#\n\x14min_translation_norm\x18\x05 \x01(\x02:\x05\x30.002\x12$\n\x17translation_blend_alpha\x18\x06 \x01(\x02:\x03\x30.7\x12\'\n\x1atranslation_prior_increase\x18\x07 \x01(\x02:\x03\x30.2*\x04\x08\x01\x10\x02\x1ay\n\x1bJointTrackEstimationOptions\x12\x1c\n\x11num_motion_models\x18\x01 \x01(\x05:\x01\x33\x12\x19\n\rmotion_stride\x18\x02 \x01(\x05:\x02\x31\x35\x12!\n\x12temporal_smoothing\x18\x03 \x01(\x08:\x05\x66\x61lse\x1a\xca\x03\n\x16LongFeatureBiasOptions\x12\x17\n\x0ctotal_rounds\x18\r \x01(\x05:\x01\x31\x12\x19\n\x0binlier_bias\x18\x01 \x01(\x02:\x04\x30.98\x12\x19\n\x0coutlier_bias\x18\x02 \x01(\x02:\x03\x30.7\x12!\n\x15num_irls_observations\x18\x03 \x01(\x05:\x02\x31\x30\x12!\n\x15max_irls_change_ratio\x18\x04 \x01(\x02:\x02\x31\x30\x12\x1f\n\x12inlier_irls_weight\x18\x05 \x01(\x02:\x03\x30.2\x12\x15\n\nbias_stdev\x18\x0c \x01(\x02:\x01\x31\x12\x1e\n\x10use_spatial_bias\x18\x06 \x01(\x08:\x04true\x12\x17\n\tgrid_size\x18\x07 \x01(\x02:\x04\x30.04\x12\x1b\n\rspatial_sigma\x18\x08 \x01(\x02:\x04\x30.02\x12\x17\n\x0b\x63olor_sigma\x18\t \x01(\x02:\x02\x32\x30\x12 \n\x14long_track_threshold\x18\n \x01(\x05:\x02\x33\x30\x12,\n\x1elong_track_confidence_fraction\x18\x0b \x01(\x02:\x04\x30.25\x12$\n\x15seed_priors_from_bias\x18\x0e \x01(\x08:\x05\x66\x61lse\x1a\xbe\x01\n\x11TranslationBounds\x12\x17\n\x0cmin_features\x18\x01 \x01(\x05:\x01\x33\x12\'\n\x19\x66rac_max_motion_magnitude\x18\x02 \x01(\x02:\x04\x30.15\x12(\n\x1amax_motion_stdev_threshold\x18\x04 \x01(\x02:\x04\x30.01\x12\x1f\n\x10max_motion_stdev\x18\x03 \x01(\x02:\x05\x30.065\x12\x1c\n\x10max_acceleration\x18\x05 \x01(\x02:\x02\x32\x30\x1a\xa6\x02\n\x10SimilarityBounds\x12\x1f\n\x11only_stable_input\x18\x01 \x01(\x08:\x04true\x12 \n\x13min_inlier_fraction\x18\x02 \x01(\x02:\x03\x30.2\x12\x17\n\x0bmin_inliers\x18\x03 \x01(\x02:\x02\x33\x30\x12\x18\n\x0blower_scale\x18\x04 \x01(\x02:\x03\x30.8\x12\x19\n\x0bupper_scale\x18\x05 \x01(\x02:\x04\x31.25\x12\x1c\n\x0elimit_rotation\x18\x06 \x01(\x02:\x04\x30.25\x12\x1b\n\x10inlier_threshold\x18\x07 \x01(\x02:\x01\x34\x12 \n\x15\x66rac_inlier_threshold\x18\x08 \x01(\x02:\x01\x30\x12$\n\x17strict_inlier_threshold\x18\t \x01(\x02:\x03\x30.5\x1a\x9d\x02\n\x10HomographyBounds\x12\x18\n\x0blower_scale\x18\x01 \x01(\x02:\x03\x30.8\x12\x19\n\x0bupper_scale\x18\x02 \x01(\x02:\x04\x31.25\x12\x1c\n\x0elimit_rotation\x18\x03 \x01(\x02:\x04\x30.25\x12!\n\x11limit_perspective\x18\x04 \x01(\x02:\x06\x30.0004\x12#\n\x16registration_threshold\x18\x05 \x01(\x02:\x03\x30.1\x12&\n\x1b\x66rac_registration_threshold\x18\x08 \x01(\x02:\x01\x30\x12 \n\x13min_inlier_coverage\x18\x06 \x01(\x02:\x03\x30.3\x12$\n\x15\x66rac_inlier_threshold\x18\x07 \x01(\x02:\x05\x30.002\x1a\xb0\x01\n\x17MixtureHomographyBounds\x12 \n\x13min_inlier_coverage\x18\x01 \x01(\x02:\x03\x30.4\x12&\n\x1bmax_adjacent_outlier_blocks\x18\x02 \x01(\x05:\x01\x35\x12$\n\x19max_adjacent_empty_blocks\x18\x03 \x01(\x05:\x01\x33\x12%\n\x15\x66rac_inlier_threshold\x18\x07 \x01(\x02:\x06\x30.0025\x1a\x95\x02\n\x17OverlayDetectionOptions\x12\x1e\n\x12\x61nalysis_mask_size\x18\x01 \x01(\x05:\x02\x31\x30\x12$\n\x17strict_near_zero_motion\x18\x02 \x01(\x02:\x03\x30.2\x12)\n\x1cstrict_max_translation_ratio\x18\x03 \x01(\x02:\x03\x30.2\x12$\n\x17strict_min_texturedness\x18\x05 \x01(\x02:\x03\x30.1\x12!\n\x16loose_near_zero_motion\x18\x04 \x01(\x02:\x01\x31\x12\x1e\n\x11overlay_min_ratio\x18\x06 \x01(\x02:\x03\x30.3\x12 \n\x14overlay_min_features\x18\x07 \x01(\x02:\x02\x31\x30\x1ar\n\x13ShotBoundaryOptions\x12*\n\x1cmotion_consistency_threshold\x18\x01 \x01(\x02:\x04\x30.02\x12/\n appearance_consistency_threshold\x18\x02 \x01(\x02:\x05\x30.075\"\x95\x01\n\x1aLinearSimilarityEstimation\x12\x16\n\x12\x45STIMATION_LS_NONE\x10\x00\x12\x14\n\x10\x45STIMATION_LS_L2\x10\x01\x12\x16\n\x12\x45STIMATION_LS_IRLS\x10\x04\x12\x1b\n\x17\x45STIMATION_LS_L2_RANSAC\x10\x02\x12\x14\n\x10\x45STIMATION_LS_L1\x10\x03\"d\n\x10\x41\x66\x66ineEstimation\x12\x1a\n\x16\x45STIMATION_AFFINE_NONE\x10\x00\x12\x18\n\x14\x45STIMATION_AFFINE_L2\x10\x01\x12\x1a\n\x16\x45STIMATION_AFFINE_IRLS\x10\x02\"e\n\x14HomographyEstimation\x12\x19\n\x15\x45STIMATION_HOMOG_NONE\x10\x00\x12\x17\n\x13\x45STIMATION_HOMOG_L2\x10\x01\x12\x19\n\x15\x45STIMATION_HOMOG_IRLS\x10\x02\"x\n\x1bMixtureHomographyEstimation\x12\x1d\n\x19\x45STIMATION_HOMOG_MIX_NONE\x10\x00\x12\x1b\n\x17\x45STIMATION_HOMOG_MIX_L2\x10\x01\x12\x1d\n\x19\x45STIMATION_HOMOG_MIX_IRLS\x10\x02\"}\n\x10\x45stimationPolicy\x12\x18\n\x14INDEPENDENT_PARALLEL\x10\x01\x12\x16\n\x12TEMPORAL_IRLS_MASK\x10\x02\x12\x1e\n\x1aTEMPORAL_LONG_FEATURE_BIAS\x10\x04\x12\x17\n\x13JOINTLY_FROM_TRACKS\x10\x03\"X\n\x10MixtureModelMode\x12\x10\n\x0c\x46ULL_MIXTURE\x10\x00\x12\x17\n\x13TRANSLATION_MIXTURE\x10\x01\x12\x19\n\x15SKEW_ROTATION_MIXTURE\x10\x02\"b\n\x10IRLSWeightFilter\x12\x14\n\x10IRLS_FILTER_NONE\x10\x00\x12\x17\n\x13IRLS_FILTER_TEXTURE\x10\x01\x12\x1f\n\x1bIRLS_FILTER_CORNER_RESPONSE\x10\x02\"\x87\x01\n\"HomographyIrlsWeightInitialization\x12\x1c\n\x18IRLS_WEIGHT_CONSTANT_ONE\x10\x01\x12\x1f\n\x1bIRLS_WEIGHT_CENTER_GAUSSIAN\x10\x02\x12\"\n\x1eIRLS_WEIGHT_PERIMETER_GAUSSIAN\x10\x03*\x04\x08\x07\x10\x08*\x04\x08\x08\x10\t*\x04\x08\x10\x10\x11')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'mediapipe.util.tracking.motion_estimation_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _MOTIONESTIMATIONOPTIONS.fields_by_name['estimate_similarity']._options = None
  _MOTIONESTIMATIONOPTIONS.fields_by_name['estimate_similarity']._serialized_options = b'\030\001'
  _globals['_MOTIONESTIMATIONOPTIONS']._serialized_start=63
  _globals['_MOTIONESTIMATIONOPTIONS']._serialized_end=6902
  _globals['_MOTIONESTIMATIONOPTIONS_IRLSOUTLIERINITIALIZATION']._serialized_start=3578
  _globals['_MOTIONESTIMATIONOPTIONS_IRLSOUTLIERINITIALIZATION']._serialized_end=3675
  _globals['_MOTIONESTIMATIONOPTIONS_LONGFEATUREINITIALIZATION']._serialized_start=3677
  _globals['_MOTIONESTIMATIONOPTIONS_LONGFEATUREINITIALIZATION']._serialized_end=3799
  _globals['_MOTIONESTIMATIONOPTIONS_IRLSMASKOPTIONS']._serialized_start=3802
  _globals['_MOTIONESTIMATIONOPTIONS_IRLSMASKOPTIONS']._serialized_end=4013
  _globals['_MOTIONESTIMATIONOPTIONS_JOINTTRACKESTIMATIONOPTIONS']._serialized_start=4015
  _globals['_MOTIONESTIMATIONOPTIONS_JOINTTRACKESTIMATIONOPTIONS']._serialized_end=4136
  _globals['_MOTIONESTIMATIONOPTIONS_LONGFEATUREBIASOPTIONS']._serialized_start=4139
  _globals['_MOTIONESTIMATIONOPTIONS_LONGFEATUREBIASOPTIONS']._serialized_end=4597
  _globals['_MOTIONESTIMATIONOPTIONS_TRANSLATIONBOUNDS']._serialized_start=4600
  _globals['_MOTIONESTIMATIONOPTIONS_TRANSLATIONBOUNDS']._serialized_end=4790
  _globals['_MOTIONESTIMATIONOPTIONS_SIMILARITYBOUNDS']._serialized_start=4793
  _globals['_MOTIONESTIMATIONOPTIONS_SIMILARITYBOUNDS']._serialized_end=5087
  _globals['_MOTIONESTIMATIONOPTIONS_HOMOGRAPHYBOUNDS']._serialized_start=5090
  _globals['_MOTIONESTIMATIONOPTIONS_HOMOGRAPHYBOUNDS']._serialized_end=5375
  _globals['_MOTIONESTIMATIONOPTIONS_MIXTUREHOMOGRAPHYBOUNDS']._serialized_start=5378
  _globals['_MOTIONESTIMATIONOPTIONS_MIXTUREHOMOGRAPHYBOUNDS']._serialized_end=5554
  _globals['_MOTIONESTIMATIONOPTIONS_OVERLAYDETECTIONOPTIONS']._serialized_start=5557
  _globals['_MOTIONESTIMATIONOPTIONS_OVERLAYDETECTIONOPTIONS']._serialized_end=5834
  _globals['_MOTIONESTIMATIONOPTIONS_SHOTBOUNDARYOPTIONS']._serialized_start=5836
  _globals['_MOTIONESTIMATIONOPTIONS_SHOTBOUNDARYOPTIONS']._serialized_end=5950
  _globals['_MOTIONESTIMATIONOPTIONS_LINEARSIMILARITYESTIMATION']._serialized_start=5953
  _globals['_MOTIONESTIMATIONOPTIONS_LINEARSIMILARITYESTIMATION']._serialized_end=6102
  _globals['_MOTIONESTIMATIONOPTIONS_AFFINEESTIMATION']._serialized_start=6104
  _globals['_MOTIONESTIMATIONOPTIONS_AFFINEESTIMATION']._serialized_end=6204
  _globals['_MOTIONESTIMATIONOPTIONS_HOMOGRAPHYESTIMATION']._serialized_start=6206
  _globals['_MOTIONESTIMATIONOPTIONS_HOMOGRAPHYESTIMATION']._serialized_end=6307
  _globals['_MOTIONESTIMATIONOPTIONS_MIXTUREHOMOGRAPHYESTIMATION']._serialized_start=6309
  _globals['_MOTIONESTIMATIONOPTIONS_MIXTUREHOMOGRAPHYESTIMATION']._serialized_end=6429
  _globals['_MOTIONESTIMATIONOPTIONS_ESTIMATIONPOLICY']._serialized_start=6431
  _globals['_MOTIONESTIMATIONOPTIONS_ESTIMATIONPOLICY']._serialized_end=6556
  _globals['_MOTIONESTIMATIONOPTIONS_MIXTUREMODELMODE']._serialized_start=6558
  _globals['_MOTIONESTIMATIONOPTIONS_MIXTUREMODELMODE']._serialized_end=6646
  _globals['_MOTIONESTIMATIONOPTIONS_IRLSWEIGHTFILTER']._serialized_start=6648
  _globals['_MOTIONESTIMATIONOPTIONS_IRLSWEIGHTFILTER']._serialized_end=6746
  _globals['_MOTIONESTIMATIONOPTIONS_HOMOGRAPHYIRLSWEIGHTINITIALIZATION']._serialized_start=6749
  _globals['_MOTIONESTIMATIONOPTIONS_HOMOGRAPHYIRLSWEIGHTINITIALIZATION']._serialized_end=6884
# @@protoc_insertion_point(module_scope)
