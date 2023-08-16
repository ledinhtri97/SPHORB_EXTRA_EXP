/*
	AUTHOR:
	Qiang Zhao, email: qiangzhao@tju.edu.cn
	Copyright (C) 2015 Tianjin University
	School of Computer Software
	School of Computer Science and Technology

	LICENSE:
	SPHORB is distributed under the GNU General Public License.  For information on 
	commercial licensing, please contact the authors at the contact address below.

	REFERENCE:
	@article{zhao-SPHORB,
	author   = {Qiang Zhao and Wei Feng and Liang Wan and Jiawan Zhang},
	title    = {SPHORB: A Fast and Robust Binary Feature on the Sphere},
	journal  = {International Journal of Computer Vision},
	year     = {2015},
	volume   = {113},
	number   = {2},
	pages    = {143-159},
	}

	
	Automatically generated code using the scheme of Rosten and Drummond
	E. Rosten and T. Drummond. Machine learning for highspeed corner detection. 
	In Proceedings of the European Conference on Computer Vision (ECCV), 2006.


	Parameters:
	splat_subtree = 1
	corner_pointers = 2
	force_first_question = 0
	corner_type = 10
	barrier = 25

	Data:
	Number of frames:    200
	Potential features:  129234000
	Real features:       2463006
	Questions per pixel: 2.63777
*/

#include "detector.h"

void sfast(Mat img, Mat imgMask, Mat geoMask, int threshold, bool nonmax_suppression,
           const int partIndex, vector<KeyPoint>& keypoints)
{
    int boundary = 18;
    int i, j, k, pixel[N];
    makeOffsets(pixel, (int)img.step, N);

    keypoints.clear();

    threshold = std::min(std::max(threshold, 0), 255);

    uchar threshold_tab[512];
    for( i = -255; i <= 255; i++ )
        threshold_tab[i+255] = (uchar)(i < -threshold ? 1 : i > threshold ? 2 : 0);

    AutoBuffer<uchar> _buf((img.cols+16)*3*(sizeof(int) + sizeof(uchar)) + 128);
    uchar* buf[3];
    buf[0] = _buf; buf[1] = buf[0] + img.cols; buf[2] = buf[1] + img.cols;
    int* cpbuf[3];
    cpbuf[0] = (int*)alignPtr(buf[2] + img.cols, sizeof(int)) + 1;
    cpbuf[1] = cpbuf[0] + img.cols + 1;
    cpbuf[2] = cpbuf[1] + img.cols + 1;
    memset(buf[0], 0, img.cols*3);

    for(i = boundary ; i < img.rows-boundary+1; i++)
    {
        const uchar* ptr = img.ptr<uchar>(i) + boundary;
        const uchar* ptr_imgMask = imgMask.ptr<uchar>(i) + boundary;
        const uchar* ptr_geoMask = geoMask.ptr<uchar>(i) + boundary;

        uchar* curr = buf[(i - boundary)%3];
        int* cornerpos = cpbuf[(i - boundary)%3];
        memset(curr, 0, img.cols);
        int ncorners = 0;

        if( i < img.rows - boundary )
        {
            j = boundary;
            for( ; j < img.cols - boundary; j++, ptr++, ptr_imgMask++, ptr_geoMask++ )
            {
                if(*ptr_imgMask==0 || *ptr_geoMask==0)
                    continue;

                int v = ptr[0];
                const uchar* tab = &threshold_tab[0] - v + 255;

                // part1, 先选六边形的3对顶点
                int d = tab[*(ptr+pixel[15])] | tab[*(ptr+pixel[6])];
                if( d == 0 )
                    continue;
                d &= tab[*(ptr+pixel[0])] | tab[*(ptr+pixel[9])];
                d &= tab[*(ptr+pixel[3])] | tab[*(ptr+pixel[12])];
                if( d == 0 )
                    continue;
                // part2, 逆时针旋转1格，约20度
                d &= tab[*(ptr+pixel[16])] | tab[*(ptr+pixel[7])];
                d &= tab[*(ptr+pixel[1])] | tab[*(ptr+pixel[10])];
                d &= tab[*(ptr+pixel[4])] | tab[*(ptr+pixel[13])];
                if( d == 0 )
                    continue;
                // part3, 逆时针旋转2格，约40度
                d &= tab[*(ptr+pixel[17])] | tab[*(ptr+pixel[8])];
                d &= tab[*(ptr+pixel[14])] | tab[*(ptr+pixel[5])];
                d &= tab[*(ptr+pixel[2])] | tab[*(ptr+pixel[11])];

                int sign_flag = 0;
                if( d & 1 )//darker，中间亮，周围暗
                    sign_flag = -1;
                else if( d & 2 )//brighter，中间暗，周围亮
                    sign_flag = 1;

                if( sign_flag == 0 )
                    continue;

                int vt = *ptr + sign_flag*threshold, count = 0;
                int sfast_score = 999; // 原来的sfastScore函数不需要了，提取角点的同时也计算分数
                for(int k = 0; k < N; k++ )
                {
                    int x = *(ptr+pixel[k]);
                    if( sign_flag*(x-vt) > 0 )
                    {
                        sfast_score = std::min(sfast_score, abs(x-*ptr));
                        if( ++count > K ) //若超过一半像素点灰度差值大于阈值，则认为是候选角点
                        {
                            cornerpos[ncorners++] = j;
                            if(nonmax_suppression)
                                curr[j] = (uchar)sfast_score;
                            break;
                        }
                    }
                    else
                    {
                        count = 0;
                        sfast_score = 999;
                    }
                }
            }
        }
        cornerpos[-1] = ncorners;

        if( i == boundary )
            continue;

        const uchar* prev = buf[(i-1)%3];
        const uchar* pprev = buf[(i-2)%3];
        cornerpos = cpbuf[(i-1)%3];
        ncorners = cornerpos[-1];

        for( k = 0; k < ncorners; k++ )
        {
            j = cornerpos[k];
            int score = prev[j];
            // 参考论文中的图4b，可以看到，领域是6个像素
            if( !nonmax_suppression ||
                (score > prev[j+1] && score > prev[j-1] &&
                 /*score > pprev[j-1] &&*/ score > pprev[j] && score > pprev[j+1] &&
                 score > curr[j-1] && score > curr[j] /*&& score > curr[j+1]*/ ) )
            {
                KeyPoint kp;
                kp.pt.x = j;
                kp.pt.y = i-1;
                kp.response = score;
                kp.class_id = partIndex;
                keypoints.push_back(kp);
            }
        }
    }
}

xy* sfast_corner_detect(const detectbyte* im, const detectbyte* im_custom_mask, const detectbyte* mask,
                        int xsize, int xstride, int ysize, int barrier, int* num)
{
    // im_custom_mask 是外部输入的自定义mask
    int boundary = 18, y;
    const detectbyte   *line_max, *line_min;
    int			rsize=512, total=0;
    xy	 		*ret = (xy*)malloc(rsize*sizeof(xy));
    const detectbyte   *pMask;
    const detectbyte* custom_mask; // custom_mask 紧跟 pMask
    const detectbyte* cache_0;

    int	pixel[N];
    makeOffsets(pixel, xstride, N);

    // from opencv 2.4 fast.cpp
    // ref line 76 : https://gitee.com/mirrors/opencv/blob/2.4/modules/features2d/src/fast.cpp
    int threshold = std::min(std::max(barrier, 0), 255);//保证阈值在0-255之间。
    uchar threshold_tab[512];
    for(int i = -255; i <= 255; i++ )
        threshold_tab[i+255] = (uchar)(i < -threshold ? 1 : i > threshold ? 2 : 0);  //分类成为darker、similar、brighter三种

    for(y = boundary ; y < ysize - boundary; y++)
    {
        cache_0 = im + boundary + y*xstride;
        pMask = mask + boundary + y*xstride;
        // custom_mask 紧跟 pMask
        custom_mask = im_custom_mask + boundary + y*xstride;

        line_min = cache_0 - boundary;
        line_max = im + xsize - boundary + y * xstride;

        // for(; cache_0 < line_max;pMask++, cache_0++, cache_1++, cache_2++)
        for(; cache_0 < line_max; pMask++, custom_mask++, cache_0++)
        {
            // if(*pMask==0)
            if(*pMask==0 || *custom_mask==0)
                continue;

            // from opencv 2.4 fast.cpp, ref line 163
            const uchar* tab = &threshold_tab[0] - *cache_0 + 255;
            // part1, 先选六边形的3对顶点
            int d = tab[*(cache_0+pixel[15])] | tab[*(cache_0+pixel[6])];
            if( d == 0 )
                continue;
            d &= tab[*(cache_0+pixel[0])] | tab[*(cache_0+pixel[9])];
            d &= tab[*(cache_0+pixel[3])] | tab[*(cache_0+pixel[12])];
            if( d == 0 )
                continue;
            // part2, 逆时针旋转1格，约20度
            d &= tab[*(cache_0+pixel[16])] | tab[*(cache_0+pixel[7])];
            d &= tab[*(cache_0+pixel[1])] | tab[*(cache_0+pixel[10])];
            d &= tab[*(cache_0+pixel[4])] | tab[*(cache_0+pixel[13])];
            if( d == 0 )
                continue;
            // part3, 逆时针旋转2格，约40度
            d &= tab[*(cache_0+pixel[17])] | tab[*(cache_0+pixel[8])];
            d &= tab[*(cache_0+pixel[14])] | tab[*(cache_0+pixel[5])];
            d &= tab[*(cache_0+pixel[2])] | tab[*(cache_0+pixel[11])];

            int sign_flag = 0;
            if( d & 1 )//darker，中间亮，周围暗
                sign_flag = -1;
            else if( d & 2 )//brighter，中间暗，周围亮
                sign_flag = 1;

            if( sign_flag == 0 )
                continue;

            int vt = *cache_0 + sign_flag*threshold, count = 0;
            for(int k = 0; k < N; k++ )
            {
                int x = *(cache_0+pixel[k]);
                if( sign_flag*(x-vt) > 0 )
                {
                    if( ++count > K ) //若超过一半像素点灰度差值大于阈值，则认为是候选角点
                    {
                        //设为候选点后，确保空间足够，并记录坐标
                        if(total >= rsize)
                        {
                            rsize *= 2;
                            ret = (xy*)realloc(ret, rsize*sizeof(xy));
                            if(ret == NULL)
                            {
                                *num=-1;
                                return NULL;
                            }
                        }
                        ret[total].x = cache_0-line_min;
                        ret[total++].y = y;
                        break;
                    }
                }
                else
                {
                    count = 0;
                }
            }
        }
    }
    *num = total;
    return ret;
}


// ref line 120, https://gitee.com/mirrors/opencv/blob/2.4/modules/features2d/src/fast_score.cpp
int sfast_corner_score(const detectbyte* im, const int pixel[], int threshold)
{
    int k, v = im[0];
    short d[N];
    for( k = 0; k < N; k++ )    //计算与周围18个像素点的差值,保存在d[k]中
        d[k] = (short)(v - im[pixel[k]]);

    int a0 = threshold;
    //这样写的目的，就是“连续、一半”的体现！
    for( k = 0; k < patternSize; k += 2 )//darker，中间亮，周围暗
    {
        int a = std::min((int)d[k+1], (int)d[k+2]);
        a = std::min(a, (int)d[k+3]);
        if( a <= a0 )
            continue;
        a = std::min(a, (int)d[k+4]);
        a = std::min(a, (int)d[k+5]);
        a = std::min(a, (int)d[k+6]);
        a = std::min(a, (int)d[k+7]);
        a = std::min(a, (int)d[k+8]);
        a = std::min(a, (int)d[k+9]);
        a0 = std::max(a0, std::min(a, (int)d[k]));
        a0 = std::max(a0, std::min(a, (int)d[k+10]));
    }
    int b0 = -a0;
    for( k = 0; k < patternSize; k += 2 )//brighter，中间暗，周围亮
    {
        int b = std::max((int)d[k+1], (int)d[k+2]);
        b = std::max(b, (int)d[k+3]);
        b = std::max(b, (int)d[k+4]);
        b = std::max(b, (int)d[k+5]);
        if( b >= b0 )
            continue;
        b = std::max(b, (int)d[k+6]);
        b = std::max(b, (int)d[k+7]);
        b = std::max(b, (int)d[k+8]);
        b = std::max(b, (int)d[k+9]);
        b0 = std::min(b0, std::max(b, (int)d[k]));
        b0 = std::min(b0, std::max(b, (int)d[k+10]));
    }
    threshold = -b0-1;

    return threshold;
}		 																			 	        
