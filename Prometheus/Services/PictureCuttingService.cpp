#include "PictureCuttingService.hpp"

#include <utility>
#include "../Modules/CUDAUtility.hpp"
#include "../Modules/ImageDebugUtility.hpp"

namespace RoboPioneers::Prometheus
{
	/// 更新方法
	void PictureCuttingService::OnUpdate(Sparrow::Frame &frame)
	{
		if (*Input.NeedToCut && !Settings.ForceNotCut)
		{
			frame.GpuPicture = CutPictureByInterestedRegion(&frame.GpuPicture,*Input.CuttingArea);
//			frame.GpuPicture = CutPictureByMask(&frame.GpuPicture, *Input.CuttingArea);
			frame.PointOffset = cv::Point{(Input.CuttingArea->x, Input.CuttingArea->y)};
//			frame.PointOffset = cv::Point{0,0};
		}
		else
		{
			frame.PointOffset = cv::Point{0,0};
		}

		#ifdef DEBUG
		frame.GpuPicture.download(frame.CutPicture);
		Modules::CUDAUtility::SynchronizeDevice();
//		if (*Input.NeedToCut)
//		{
//			frame.CutPicture = frame.CutPicture(*Input.CuttingArea);
//		}
		cv::imshow("Cutting Picture", frame.CutPicture);
		#endif
	}

	/// 使用兴趣区方式裁剪图像
	cv::cuda::GpuMat PictureCuttingService::CutPictureByInterestedRegion(cv::cuda::GpuMat const *picture, cv::Rect area)
	{
//		cv::cuda::GpuMat target(area.height, area.width, picture->type());
//		(*picture)(std::move(area)).copyTo(target);
//		Modules::CUDAUtility::SynchronizeDevice();
//		return target;

		return (*picture)(area).clone();
	}
	/// 使用蒙版的方式裁剪图像
	cv::cuda::GpuMat PictureCuttingService::CutPictureByMask(const cv::cuda::GpuMat *picture, cv::Rect area)
	{
		cv::cuda::GpuMat target(picture->rows, picture->cols, picture->type(),cv::Scalar(0,0,0));
		(*picture)(area).copyTo(target(area));
		Modules::CUDAUtility::SynchronizeDevice();
		return target;
	}
}
