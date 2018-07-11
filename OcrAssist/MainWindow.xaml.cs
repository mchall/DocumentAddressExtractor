using Microsoft.Win32;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Imaging;

namespace OcrAssist
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : System.Windows.Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        public void Test()
        {
            OpenFileDialog ofd = new OpenFileDialog();
            ofd.Filter = "Image|*.jpg;*.png";
            if (ofd.ShowDialog() == true)
            {
                Mat color = new Mat(ofd.FileName, ImreadModes.Color);
                Mat src = new Mat(ofd.FileName, ImreadModes.GrayScale);

                Mat thresh = new Mat();
                Cv2.AdaptiveThreshold(src, thresh, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.Binary, 25, 10);
                Cv2.BitwiseNot(thresh, thresh);

                /*var vStructure = Cv2.GetStructuringElement(MorphShapes.Rect, new OpenCvSharp.Size(1, src.Rows / 30));
                Mat vertical = new Mat();
                Cv2.Erode(thresh, vertical, vStructure, new OpenCvSharp.Point(-1, -1));
                Cv2.Dilate(vertical, vertical, vStructure, new OpenCvSharp.Point(-1, -1)); 

                var hStructure = Cv2.GetStructuringElement(MorphShapes.Rect, new OpenCvSharp.Size(src.Cols / 30, 1));
                Mat horizontal = new Mat();
                Cv2.Erode(thresh, horizontal, hStructure, new OpenCvSharp.Point(-1, -1));
                Cv2.Dilate(horizontal, horizontal, hStructure, new OpenCvSharp.Point(-1, -1)); 

                Mat add = new Mat();
                Cv2.AddWeighted(vertical, 255, horizontal, 255, 0, add);
                Mat scaled = thresh - add;*/

                Mat scaled = src.Clone();
                //Cv2.PyrDown(scaled, scaled);
                //Cv2.PyrUp(scaled, scaled);

                Mat grad = new Mat();
                //var morphKernel = Cv2.GetStructuringElement(MorphShapes.Ellipse, new OpenCvSharp.Size(3, 3));
                var morphKernel = Cv2.GetStructuringElement(MorphShapes.Rect, new OpenCvSharp.Size(16, 1));
                Cv2.MorphologyEx(scaled, grad, MorphTypes.Gradient, morphKernel);

                Mat bw = new Mat();
                Cv2.Threshold(grad, bw, 32, 255, ThresholdTypes.Binary | ThresholdTypes.Otsu);

                Mat connected = new Mat();
                morphKernel = Cv2.GetStructuringElement(MorphShapes.Rect, new OpenCvSharp.Size(16, 1));
                Cv2.MorphologyEx(bw, connected, MorphTypes.Close, morphKernel);

                OpenCvSharp.Point[][] contours;
                HierarchyIndex[] hierarchy;
                Cv2.FindContours(connected, out contours, out hierarchy, RetrievalModes.CComp, ContourApproximationModes.ApproxSimple);

                var rectangles = new List<OpenCvSharp.Rect>();
                for (int i = 0; i < hierarchy.Length; i++)
                {
                    rectangles.Add(Cv2.BoundingRect(contours[i]));
                }

                var merged = MergeRects(rectangles.ToArray(), src.Width, src.Height);
                for (int i = 0; i < merged.Length; i++)
                {
                    Mat roi = new Mat(thresh, merged[i]);
                    Mat rowReduce = new Mat();
                    Mat colReduce = new Mat();
                    Cv2.Reduce(roi, rowReduce, ReduceDimension.Row, ReduceTypes.Avg, MatType.CV_32S);
                    Cv2.Reduce(roi, colReduce, ReduceDimension.Column, ReduceTypes.Avg, MatType.CV_32S);

                    //todo: segments >= 2 ?? must be black on left+right
                    //todo: vertical, must be black on top+bottom
                    //todo: count of whole image ????

                    var rowPercent = Math.Round((double)Cv2.CountNonZero(rowReduce) / rowReduce.Cols, 2);
                    var colPercent = Math.Round((double)Cv2.CountNonZero(colReduce) / colReduce.Rows, 2);
                    if (rowPercent > 0.6 && rowPercent < 0.95)
                    {
                        Cv2.Rectangle(color, merged[i], Scalar.Green, 4);
                    }
                    else
                    {
                        Cv2.Rectangle(color, merged[i], Scalar.Red, 4);
                    }
                }

                MemoryStream ms = new MemoryStream();
                color.WriteToStream(ms);
                var imageSource = new BitmapImage();
                imageSource.BeginInit();
                imageSource.StreamSource = ms;
                imageSource.EndInit();
                Image.Source = imageSource;

                MemoryStream ms2 = new MemoryStream();
                thresh.WriteToStream(ms2);
                var imageSource2 = new BitmapImage();
                imageSource2.BeginInit();
                imageSource2.StreamSource = ms2;
                imageSource2.EndInit();
                Original.Source = imageSource2;
            }
        }

        private OpenCvSharp.Rect[] MergeRects(OpenCvSharp.Rect[] rects, int width, int height)
        {
            List<int> ignoreIndices = new List<int>();

            for (int i = 0; i < rects.Length; i++)
            {
                if (ignoreIndices.Contains(i))
                    continue;

                if (rects[i].Width + rects[i].X > width)
                    rects[i].Width = width - rects[i].X;

                if (rects[i].Height + rects[i].Y > height)
                    rects[i].Height = height - rects[i].Y;

                for (int j = i + 1; j < rects.Length; j++)
                {
                    if (i == j)
                        continue;

                    if (rects[i].Contains(rects[j]))
                    {
                        ignoreIndices.Add(j);
                    }
                }
            }

            var merged = new List<OpenCvSharp.Rect>();
            for (int i = 0; i < rects.Length; i++)
            {
                if (!ignoreIndices.Contains(i))
                {
                    merged.Add(rects[i]);
                }
            }
            return merged.ToArray();
        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            Test();
        }
    }
}