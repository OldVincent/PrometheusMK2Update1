#include "BattleIntelligenceService.hpp"

#include "../Modules/ImageDebugUtility.hpp"
#include "../Modules/MathUtility.hpp"
#include "../Modules/GeometryFeatureModule.hpp"
#include <iostream>

namespace RoboPioneers::Prometheus
{
	#ifdef DEBUG
	cv::Mat DebugPictureIntelligence;
	#endif

	void BattleIntelligenceService::OnUpdate(Sparrow::Frame &frame)
	{
		#ifdef DEBUG
		DebugPictureIntelligence = frame.CutPicture.clone();
		#endif

		bool found = false;
		cv::RotatedRect best_one;

		double best_one_score = 0;
		ElementPair best_pair;
		if (!Input.PossibleArmors->empty())
		{
			for (const auto& candidate : *Input.PossibleArmors)
			{
				auto current_rectangle = CastPairToRotatedRectangle(candidate);

				auto geometry_parameters = Modules::GeometryFeatureModule::StandardizeRotatedRectangle(current_rectangle);

				constexpr double standard_big_armor_aspect_ratio = 23.5f / 6.0f;
				constexpr double standard_small_armor_aspect_ratio = 14.0f / 6.0f;

				double small_armor_ratio = Modules::MathUtility::ResembleCoefficient(
						geometry_parameters.Length / geometry_parameters.Width, standard_small_armor_aspect_ratio);

				double big_armor_ratio = Modules::MathUtility::ResembleCoefficient(
						geometry_parameters.Length / geometry_parameters.Width, standard_big_armor_aspect_ratio
				);
				if (big_armor_ratio < 0.4 && small_armor_ratio < 0.4)
				{
					continue;
				}

				double score = (small_armor_ratio > big_armor_ratio ? small_armor_ratio : big_armor_ratio) *
				               Modules::MathUtility::ResembleCoefficient(geometry_parameters.Angle, 90) *
				               geometry_parameters.Length * geometry_parameters.Width;

				if (score >= best_one_score)
				{
					best_pair = candidate;
					best_one = current_rectangle;
					found = true;
				}
			}
		}

		auto best_geometry = Modules::GeometryFeatureModule::StandardizeRotatedRectangle(best_one);
		auto best_one_global = GetGlobalRectangle(best_geometry, frame.PointOffset);

		if (found)
		{
			if ((Output.Tracked && IsSameArmor(best_one, LastTarget) || !Output.Tracked))
			{
				constexpr double width_expand = 2.0f;
				constexpr double height_expand = 2.0f;
				auto length = best_one_global.Length;
				auto width = best_one_global.Width;
				if (length < 80) length = 120;
				if (width < 80) width = 80;
				Output.InterestedArea = cv::Rect(cv::Point(
						best_one_global.Center.x - 0.5 * (width_expand) * length,
						best_one_global.Center.y - 0.5 * (height_expand) * width),
					    cv::Size(length * width_expand, width * height_expand));

				if (Output.InterestedArea.x < 0) Output.InterestedArea.x = 0;
				if (Output.InterestedArea.x > 1280) Output.InterestedArea.x = 1280;
				if (Output.InterestedArea.y < 0) Output.InterestedArea.y = 0;
				if (Output.InterestedArea.y > 1024) Output.InterestedArea.y = 1024;
				if (Output.InterestedArea.x + Output.InterestedArea.width > 1280) Output.InterestedArea.width = 1280 - Output.InterestedArea.x;
				if (Output.InterestedArea.y + Output.InterestedArea.height > 1024) Output.InterestedArea.height = 1024 - Output.InterestedArea.y;

//				Output.InterestedArea = best_one.boundingRect();
//				Output.InterestedArea = Modules::MathUtility::ScaleRectangle(Output.InterestedArea, {1.0f, 1.0f},
//				                                                            cv::Size(1280, 1024));
//				Output.InterestedArea = Modules::MathUtility::ScaleRectangle(Output.InterestedArea, {1.5f, 2.0f});
				Output.Tracked = true;
				Output.X = best_one_global.Center.x;
				Output.Y = best_one_global.Center.y;
				LastTarget = best_one;
				TrackingRemainTimes = Settings.TrackingFrames;

				std::cout << "Found X:" << Output.X  << " Y:" << Output.Y << std::endl;
			}
			else
			{
				found = false;
			}
		}

		if (!found)
		{
			if (Output.Tracked && TrackingRemainTimes >= 0)
			{
				--TrackingRemainTimes;
//				Output.InterestedArea = Modules::MathUtility::ScaleRectangle(Output.InterestedArea, {1.2f, 1.2f},
//				                                                             cv::Size(1280, 1024));
			}
			else
			{
				Output.Tracked = false;
			}
		}

		#ifdef DEBUG
		if (found)
		{
			best_one.center = best_one.center + Modules::MathUtility::ChangePointType<float>(frame.PointOffset);
			Modules::ImageDebugUtility::DrawRotatedRectangle(DebugPictureIntelligence, best_one,
			                                                 cv::Scalar(0, 255, 255));
		}
		cv::imshow("Final Decision", DebugPictureIntelligence);

		#endif
	}

	/// 从灯条对匹配旋转矩形
	cv::RotatedRect BattleIntelligenceService::CastPairToRotatedRectangle(
			const BattleIntelligenceService::ElementPair &pair)
	{
		std::vector<cv::Point> mixed_contour;
		mixed_contour.reserve(std::get<0>(pair).Raw.Contour.size() + std::get<1>(pair).Raw.Contour.size());
		mixed_contour.insert(mixed_contour.end(), std::get<0>(pair).Raw.Contour.begin(), std::get<0>(pair).Raw.Contour.end());
		mixed_contour.insert(mixed_contour.end(), std::get<1>(pair).Raw.Contour.begin(), std::get<1>(pair).Raw.Contour.end());
		return cv::minAreaRect(mixed_contour);
	}

	/// 判断是否是同一块装甲板
	bool BattleIntelligenceService::IsSameArmor(const cv::RotatedRect& current_target, const cv::RotatedRect& previous_target) const
	{
		if (Modules::MathUtility::ResembleCoefficient(current_target.size.area(), previous_target.size.area()) < 0.5)
		{
			return false;
		}
		std::vector<cv::Point> intersection_region {};
		auto intersection_type = cv::rotatedRectangleIntersection(current_target, previous_target, intersection_region);
		switch (intersection_type)
		{
			default:
			case cv::RectanglesIntersectTypes::INTERSECT_NONE:
				return false;
				break;
			case cv::RectanglesIntersectTypes::INTERSECT_FULL:
				return true;
			case cv::RectanglesIntersectTypes::INTERSECT_PARTIAL:
				auto intersection_area = cv::contourArea(intersection_region);
				// 要求重叠面积占比达标
				if (intersection_area / previous_target.size.area() < Settings.IntersectionAreaRatioThreshold)
					return false;
//				// 要求角度近乎平行
//				auto geometry_current_target =
//						Modules::GeometryFeatureModule::StandardizeRotatedRectangle(current_target);
//				auto geometry_previous_target =
//						Modules::GeometryFeatureModule::StandardizeRotatedRectangle(previous_target);
//				if (Modules::MathUtility::ResembleCoefficient(geometry_current_target.Angle, geometry_previous_target.Angle)
//				< Settings.AngleRatioThreshold)
//					return false;

				return true;;
		}
	}
}