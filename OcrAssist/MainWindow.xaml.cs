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
using System.Text;

namespace OcrAssist
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : System.Windows.Window
    {
        private TesseractEngine _ocr;

        public MainWindow()
        {
            InitializeComponent();
            _ocr = new TesseractEngine("./tessdata", "eng", EngineMode.Default);
        }

        private string TryOcr(byte[] buffer)
        {
            var pix = Pix.LoadTiffFromMemory(buffer);
            using (var page = _ocr.Process(pix, PageSegMode.SingleBlock))
            {
                return page.GetText();
            }
        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog();
            ofd.Filter = "Image|*.jpg;*.png";
            if (ofd.ShowDialog() == true)
            {
                Title = ofd.FileName;
                TryOCR(ofd.FileName);
            }
        }

        private void BulkTest()
        {
            var folder = @"C:\Users\Mike\Desktop\text recognition";
            foreach (var file in Directory.EnumerateFiles(folder))
            {
                var output = TryOCR(file);
                Cv2.ImWrite("out\\" + Path.GetFileName(file) + ".jpg", output);
            }
        }

        private Mat TryOCR(string fileName)
        {
            Mat debug = new Mat(fileName, ImreadModes.Color);
            Mat src = debug.CvtColor(ColorConversionCodes.RGB2GRAY);

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

            Mat grad = new Mat();
            var expand = ((int)Math.Round(src.Width * 0.02, 0));
            var morphKernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Cv.Size(expand, 1));
            Cv2.MorphologyEx(src, grad, MorphTypes.Gradient, morphKernel);

            Mat bw = new Mat();
            Cv2.Threshold(grad, bw, 32, 255, ThresholdTypes.Binary | ThresholdTypes.Otsu);

            Cv.Point[][] contours;
            HierarchyIndex[] hierarchy;
            Cv2.FindContours(bw, out contours, out hierarchy, RetrievalModes.CComp, ContourApproximationModes.ApproxSimple);

            var rectangles = new List<Cv.Rect>();
            for (int i = 0; i < hierarchy.Length; i++)
            {
                var boundingRect = Cv2.BoundingRect(contours[i]);
                if (!IsTooLarge(boundingRect, src) && !IsTooSmall(boundingRect, src))
                {
                    rectangles.Add(boundingRect);
                }
            }

            //TODO: cut to get rid of left/right margin

            var merged = MergeRects(rectangles, src.Width, src.Height);

            List<Cv.Rect> filtered = new List<Cv.Rect>();
            for (int i = 0; i < merged.Count; i++)
            {
                Mat roi = new Mat(thresh, merged[i]);

                if (HeuristicCheck(roi))
                {
                    filtered.Add(merged[i]);
                }
            }

            filtered = ZoneOfInterest(filtered, debug);

            var grouped = GroupRects(filtered, debug);

            StringBuilder sb = new StringBuilder();

            foreach (var group in grouped)
            {
                if (group.Count > 1 && group.Count < 6)
                {
                    var groupRect = group.First();
                    foreach (var rect in group)
                    {
                        groupRect = groupRect.Union(rect);
                    }

                    Cv2.Rectangle(debug, groupRect, Scalar.Purple, 2);

                    Mat groupRoi = new Mat(src, groupRect);

                    byte[] buff;
                    if (Cv2.ImEncode(".tiff", groupRoi, out buff))
                    {
                        sb.AppendLine(TryOcr(buff));
                        sb.AppendLine();
                    }
                }
            }

            ResultText.Text = sb.ToString();

            //Cv2.ImWrite("ocr.tiff", ocr);

            MemoryStream ms = new MemoryStream();
            debug.WriteToStream(ms);
            var imageSource = new BitmapImage();
            imageSource.BeginInit();
            imageSource.StreamSource = ms;
            imageSource.EndInit();
            Image.Source = imageSource;

            return debug;
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

            var segments = CountSegements(values);
            if (segments < 2)
            {
                return false;
            }

            return true;
        }

        private bool HeuristicColumnCheck(Mat colReduce)
        {
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

        private List<Cv.Rect> ZoneOfInterest(List<Cv.Rect> input, Mat debug)
        {
            List<Cv.Rect> output = new List<Cv.Rect>();

            for (int i = 0; i < input.Count; i++)
            {
                var l = new Cv.Rect(input[i].X + input[i].Width, input[i].Y, input[i].Width / 2, 2);
                var r = new Cv.Rect(input[i].X - (input[i].Width / 2), input[i].Y, input[i].Width / 2, 2);

                int numIntersection = 0;
                for (int j = 0; j < input.Count; j++)
                {
                    if (i == j)
                        continue;

                    if (l.IntersectsWith(input[j]) || r.IntersectsWith(input[j]))
                    {
                        numIntersection++;
                    }
                }

                if (numIntersection < 2)
                {
                    output.Add(input[i]);
                }
            }
            return output;
        }

        private bool IsTooLarge(Cv.Rect r, Mat image)
        {
            return (r.Width > image.Width * 0.4 || r.Height > image.Height * 0.2);
        }

        private bool IsTooSmall(Cv.Rect r, Mat image)
        {
            return (r.Width < image.Width * 0.04 || r.Height < image.Height * 0.005);
        }

        private List<List<Cv.Rect>> GroupRects(List<Cv.Rect> rects, Mat debug)
        {
            rects.Sort((l, r) => l.Y.CompareTo(r.Y));

            List<List<Cv.Rect>> grouped = new List<List<Cv.Rect>>();
            List<int> ignoreIndices = new List<int>();

            for (int i = 0; i < rects.Count; i++)
            {
                if (ignoreIndices.Contains(i))
                    continue;

                var current = rects[i];

                if (IsTooLarge(current, debug))
                    continue;

                List<Cv.Rect> group = new List<Cv.Rect>();
                group.Add(rects[i]);

                for (int j = i + 1; j < rects.Count; j++)
                {
                    if (IsTooLarge(rects[j], debug))
                    {
                        ignoreIndices.Add(j);
                        continue;
                    }

                    if (rects[j].Y > current.Y + current.Height + (rects[j].Height * 2))
                        continue;

                    var l = new Cv.Rect(rects[i].X, 0, rects[i].Width, rects[i].Height);
                    var r = new Cv.Rect(rects[j].X, 0, rects[j].Width, rects[j].Height);

                    if (l.IntersectsWith(r))
                    {
                        var intersect = l.Intersect(r);
                        var intersectArea = intersect.Width * intersect.Height;
                        var lArea = rects[i].Width * rects[i].Height;
                        var rArea = rects[j].Width * rects[j].Height;

                        if (intersectArea > (lArea * 0.4) && intersectArea > (rArea * 0.4))
                        {
                            current = current.Union(rects[j]);
                            group.Add(rects[j]);
                        }
                    }
                }

                grouped.Add(group);
            }

            return grouped;
        }

        private List<Cv.Rect> MergeRects(List<Cv.Rect> rects, int width, int height)
        {
            List<int> ignoreIndices = new List<int>();

            rects.Sort((l, r) => l.X.CompareTo(r.X));

            for (int i = 0; i < rects.Count; i++)
            {
                if (ignoreIndices.Contains(i))
                    continue;

                for (int j = i + 1; j < rects.Count; j++)
                {
                    if (rects[j].Height < 5 || rects[j].Width < 5)
                    {
                        ignoreIndices.Add(j);
                        continue;
                    }

                    if (rects[i].IntersectsWith(rects[j]))
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
 