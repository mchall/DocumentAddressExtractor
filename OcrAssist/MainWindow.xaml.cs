using System;
using System.Collections.Generic;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Imaging;
using Microsoft.Win32;
using OpenCvSharp;
using System.Linq;
using Tesseract;
using Cv = OpenCvSharp;

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
            OcrTest();
        }

        private void OcrTest()
        {
            TesseractEngine engine = new TesseractEngine("./tessdata", "eng", EngineMode.Default);

            var pix = Pix.LoadFromFile(@"ocr.tiff");
            using (var page = engine.Process(pix))
            {
                var text = page.GetText();
            }
        }

        private void Button_Click(object sender, RoutedEventArgs e)
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

                /*var vStructure = Cv2.GetStructuringElement(MorphShapes.Rect, new Cv.Size(1, src.Rows / 50));
                Mat vertical = new Mat();
                Cv2.Erode(thresh, vertical, vStructure, new Cv.Point(-1, -1));
                Cv2.Dilate(vertical, vertical, vStructure, new Cv.Point(-1, -1)); 

                var hStructure = Cv2.GetStructuringElement(MorphShapes.Rect, new Cv.Size(src.Cols / 50, 1));
                Mat horizontal = new Mat();
                Cv2.Erode(thresh, horizontal, hStructure, new Cv.Point(-1, -1));
                Cv2.Dilate(horizontal, horizontal, hStructure, new Cv.Point(-1, -1)); 

                Mat add = new Mat();
                Cv2.AddWeighted(vertical, 255, horizontal, 255, 0, add);
                //var dilateElement = Cv2.GetStructuringElement(MorphShapes.Rect, new Cv.Size(src.Cols / 50, src.Rows / 50));
                //Cv2.Dilate(add, add, dilateElement);
                Mat scaled = thresh - add;*/

                Mat scaled = src.Clone();
                //Cv2.PyrDown(scaled, scaled);
                //Cv2.PyrUp(scaled, scaled);

                Mat grad = new Mat();
                //var morphKernel = Cv2.GetStructuringElement(MorphShapes.Ellipse, new Cv.Size(3, 3));
                var morphKernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Cv.Size(16, 1)); //todo: based on width
                Cv2.MorphologyEx(scaled, grad, MorphTypes.Gradient, morphKernel);

                Mat bw = new Mat();
                Cv2.Threshold(grad, bw, 32, 255, ThresholdTypes.Binary | ThresholdTypes.Otsu);

                Cv.Point[][] contours;
                HierarchyIndex[] hierarchy;
                Cv2.FindContours(bw, out contours, out hierarchy, RetrievalModes.CComp, ContourApproximationModes.ApproxSimple);

                var rectangles = new List<Cv.Rect>();
                for (int i = 0; i < hierarchy.Length; i++)
                {
                    rectangles.Add(Cv2.BoundingRect(contours[i]));
                }

                var hExpand = (int)Math.Round(src.Width * 0.01, 0);
                var merged = MergeRects(rectangles, src.Width, src.Height, hExpand);

                List<Cv.Rect> filtered = new List<Cv.Rect>();
                for (int i = 0; i < merged.Count; i++)
                {
                    Mat roi = new Mat(thresh, merged[i]);

                    if (HeuristicCheck(roi))
                    {
                        filtered.Add(merged[i]);
                    }
                    else
                    {
                        Cv2.Rectangle(color, merged[i], Scalar.Red, 4);
                    }
                }

                var grouped = GroupRects(filtered);

                foreach (var group in grouped)
                {
                    var rect = group.First();
                    foreach (var r in group)
                    {
                        rect = rect.Union(r);
                    }
                    Cv2.Rectangle(color, rect, Scalar.Purple, 6);
                }

                Mat ocr = new Mat(thresh.Rows, thresh.Cols, thresh.Type());
                foreach(var group in grouped)
                {
                    bool tooLarge = group.Exists(g => g.Width > src.Width * 0.5 || g.Height > src.Height * 0.25);

                    if (!tooLarge && group.Count > 1 && group.Count < 6)
                    {
                        foreach (var rect in group)
                        {
                            Cv2.Rectangle(color, rect, Scalar.Green, 4);

                            Mat roi = new Mat(thresh, rect);
                            Mat ocrRoi = new Mat(ocr, rect);
                            roi.CopyTo(ocrRoi);
                        }
                    }
                    else
                    {
                        foreach (var rect in group)
                        {
                            Cv2.Rectangle(color, rect, Scalar.Orange, 4);
                        }
                    }
                }

                //Cv2.ImWrite("ocr.tiff", ocr);
                //OcrTest();

                MemoryStream ms = new MemoryStream();
                color.WriteToStream(ms);
                var imageSource = new BitmapImage();
                imageSource.BeginInit();
                imageSource.StreamSource = ms;
                imageSource.EndInit();
                Image.Source = imageSource;

                MemoryStream ms2 = new MemoryStream();
                ocr.WriteToStream(ms2);
                var imageSource2 = new BitmapImage();
                imageSource2.BeginInit();
                imageSource2.StreamSource = ms2;
                imageSource2.EndInit();
                Original.Source = imageSource2;
            }
        }

        private bool HeuristicCheck(Mat roi)
        {
            /*var erosion = Cv2.GetStructuringElement(MorphShapes.Rect, new Cv.Size(2, 2));
            Mat erodedRoi = new Mat();
            Cv2.Erode(roi, erodedRoi, erosion);*/

            Mat rowReduce = new Mat();
            Mat colReduce = new Mat();
            Cv2.Reduce(roi, rowReduce, ReduceDimension.Row, ReduceTypes.Sum, MatType.CV_32S);
            Cv2.Reduce(roi, colReduce, ReduceDimension.Column, ReduceTypes.Sum, MatType.CV_32S);

            //var totPercent = Math.Round((double)Cv2.CountNonZero(roi) / (roi.Width * roi.Height), 2);

            //todo: count of whole image ????
            //todo: min height/width

            /*Cv2.ImWrite("roi.tiff", roi);
            Cv2.ImWrite("roi-row.tiff", rowReduce);
            Cv2.ImWrite("roi-col.tiff", colReduce);*/

            if (!HeuristicRowCheck(rowReduce))
            {
                return false;
            }

            if (!HeuristicColumnCheck(colReduce))
            {
                return false;
            }

            return true;
        }

        private bool HeuristicRowCheck(Mat rowReduce)
        {
            var rowPercent = Math.Round((double)Cv2.CountNonZero(rowReduce) / rowReduce.Cols, 2);

            if (rowPercent < 0.55 || rowPercent > 0.95)
            {
                return false;
            }

            var csv = rowReduce.Dump(DumpFormat.Csv);
            var values = csv.Replace("[", "").Replace("]", "").Split(',').ToList().ConvertAll(s => int.Parse(s) > 0 ? 1 : 0);

            /*if (values.First() != 0 || values.Last() != 0)
            {
                return false;
            }*/

            var segments = CountSegements(values);
            if (segments < 2)
            {
                return false;
            }

            return true;
        }

        private bool HeuristicColumnCheck(Mat colReduce)
        {
            //var colPercent = Math.Round((double)Cv2.CountNonZero(colReduce) / colReduce.Rows, 2);

            var csv = colReduce.Dump(DumpFormat.Csv);
            var values = csv.Replace("[", "").Replace("]", "").Replace("\n", "").Split(';').ToList().ConvertAll(s => int.Parse(s) > 0 ? 1 : 0);

            var segments = CountSegements(values);
            if (segments > 1)
            {
                return false;
            }

            return true;
        }

        private int CountSegements(List<int> values)
        {
            int segmentCount = 0;
            bool prevWasZero = false;

            foreach (var val in values)
            {
                if (val == 0 && !prevWasZero)
                {
                    segmentCount++;
                    prevWasZero = true;
                }
                else if (val > 0)
                {
                    prevWasZero = false;
                }
            }

            return segmentCount - (prevWasZero ? 1 : 0);
        }

        private List<List<Cv.Rect>> GroupRects(List<Cv.Rect> rects)
        {
            rects.Sort((l, r) => l.Y.CompareTo(r.Y));

            List<List<Cv.Rect>> grouped = new List<List<Cv.Rect>>();
            List<int> ignoreIndices = new List<int>();

            for (int i = 0; i < rects.Count; i++)
            {
                if (ignoreIndices.Contains(i))
                    continue;

                var current = rects[i];

                List<Cv.Rect> group = new List<Cv.Rect>();
                group.Add(rects[i]);

                for (int j = i + 1; j < rects.Count; j++)
                {
                    if (i == j)
                        continue;

                    //var l = new Cv.Rect(current.X, current.Y, current.Width, (int)(current.Height + 15));
                    var l = new Cv.Rect(current.X, current.Y, current.Width, current.Height + (int)(rects[i].Height * 0.75));
                    var r = new Cv.Rect(rects[j].X, rects[j].Y, rects[j].Width, rects[j].Height);

                    //TODO: x intersection - need significant overlap

                    if (l.IntersectsWith(r))
                    {
                        current = current.Union(rects[j]);

                        ignoreIndices.Add(j);
                        group.Add(rects[j]);
                    }
                }

                grouped.Add(group);
            }

            return grouped;
        }

        private List<Cv.Rect> MergeRects(List<Cv.Rect> rects, int width, int height, int hInflate = 0, int vInflate = 0, int iterations = 2)
        {
            List<int> ignoreIndices = new List<int>();

            rects.Sort((l, r) => l.X.CompareTo(r.X));

            for (int x = 0; x < iterations; x++)
            {
                for (int i = 0; i < rects.Count; i++)
                {
                    if (ignoreIndices.Contains(i))
                        continue;

                    for (int j = i + 1; j < rects.Count; j++)
                    {
                        if (i == j)
                            continue;

                        var l = new Cv.Rect(rects[i].X, rects[i].Y, rects[i].Width, rects[i].Height);
                        var r = new Cv.Rect(rects[j].X, rects[j].Y, rects[j].Width, rects[j].Height);
                        l.Inflate(hInflate / 2, vInflate / 2);
                        r.Inflate(hInflate / 2, vInflate / 2);

                        if (l.IntersectsWith(r))
                        {
                            var union = rects[i].Union(rects[j]);

                            if (union.Width < width * 0.30 && union.Height < height * 0.05)
                            {
                                rects[i] = union;
                                ignoreIndices.Add(j);
                            }
                        }
                    }
                }
            }

            var merged = new List<Cv.Rect>();
            for (int i = 0; i < rects.Count; i++)
            {
                FixInvalidRects(rects[i], width, height);

                if (!ignoreIndices.Contains(i))
                {
                    merged.Add(rects[i]);
                }
            }
            return merged;
        }

        private static void FixInvalidRects(Cv.Rect rect, int width, int height)
        {
            if (rect.X < 0)
                rect.X = 0;

            if (rect.Y < 0)
                rect.Y = 0;

            if (rect.X > width)
                rect.Width = width;

            if (rect.Y > height)
                rect.Height = height;

            if (rect.Width + rect.X > width)
                rect.Width = width - rect.X;

            if (rect.Height + rect.Y > height)
                rect.Height = height - rect.Y;
        }
    }
}