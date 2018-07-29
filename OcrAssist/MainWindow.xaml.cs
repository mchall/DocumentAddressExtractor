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
using edu.stanford.nlp.ie.crf;
using System.Threading.Tasks;
using edu.stanford.nlp.util;

namespace OcrAssist
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : System.Windows.Window
    {
        private TesseractEngine _ocr;
        private CRFClassifier _classifier;

        public MainWindow()
        {
            InitializeComponent();
            Load3rdParty();
        }

        private async Task Load3rdParty()
        {
            Loading.Visibility = Visibility.Visible;
            _ocr = new TesseractEngine("./tessdata", "eng", EngineMode.Default);
            _classifier = await Task.Run(() => CRFClassifier.getClassifierNoExceptions(@"english.all.3class.distsim.crf.ser.gz"));
            RunButton.Content = "Run";
            RunButton.IsEnabled = true;
            Loading.Visibility = Visibility.Collapsed;
        }

        private async void Button_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog();
            ofd.Filter = "Image|*.jpg;*.png";
            if (ofd.ShowDialog() == true)
            {
                Loading.Visibility = Visibility.Visible;
                RunButton.Content = "Crunching...";
                RunButton.IsEnabled = false;

                var output = await Task.Run(() => TryOCR(ofd.FileName));

                MemoryStream ms = new MemoryStream();
                output.DebugImage.WriteToStream(ms);
                var imageSource = new BitmapImage();
                imageSource.BeginInit();
                imageSource.StreamSource = ms;
                imageSource.EndInit();
                Image.Source = imageSource;

                ResultText.Text = output.OcrText;

                RunButton.Content = "Run";
                RunButton.IsEnabled = true;
                Loading.Visibility = Visibility.Collapsed;
            }
        }

        /*private void BulkTest()
        {
            var folder = @"C:\Users\Mike\Desktop\text recognition";
            foreach (var file in Directory.EnumerateFiles(folder))
            {
                var output = TryOCR(file);
                Cv2.ImWrite("out\\" + Path.GetFileName(file) + ".jpg", output.DebugImage);
            }
        }*/

        private bool TryOcrAddress(byte[] buffer, out string output)
        {
            output = string.Empty;

            var pix = Pix.LoadTiffFromMemory(buffer);
            using (var page = _ocr.Process(pix, PageSegMode.SingleBlock))
            {
                var pageText = page.GetText();
                var lines = pageText.Split(new string[2] { "\n", "\r" }, StringSplitOptions.RemoveEmptyEntries);
                var flat = String.Join("\n", lines);

                if (lines.Length < 2)
                {
                    return false;
                }

                if (lines[0].Any(c => Char.IsDigit(c)))
                {
                    return false;
                }

                if (!lines[1].Any(c => Char.IsDigit(c)))
                {
                    return false;
                }

                if (!lines[1].Any(c => Char.IsLetter(c)))
                {
                    return false;
                }

                if (lines.Length > 2 && !lines[2].Any(c => Char.IsDigit(c)))
                {
                    return false;
                }

                if (lines.Length > 2 && !lines[2].Any(c => Char.IsLetter(c)))
                {
                    return false;
                }

                bool hasPerson = false;
                bool hasLocation = false;
                foreach (Triple result in _classifier.classifyToCharacterOffsets(lines[0]).toArray())
                {
                    hasPerson |= result.first().ToString() == "PERSON";
                    hasLocation |= result.first().ToString() == "LOCATION";
                }

                if (hasPerson)
                {
                    output = flat;
                    return true;
                }
                return false;
            }
        }

        private OcrResult TryOCR(string fileName)
        {
            Mat debug = new Mat(fileName, ImreadModes.Color);
            Mat src = debug.CvtColor(ColorConversionCodes.RGB2GRAY);

            Mat thresh = new Mat();
            Cv2.AdaptiveThreshold(src, thresh, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.Binary, 25, 10);
            Cv2.BitwiseNot(thresh, thresh);

            Mat grad = new Mat();
            var expand = ((int)Math.Round(src.Width * 0.025, 0));
            var morphKernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Cv.Size(expand, 1));
            Cv2.MorphologyEx(src, grad, MorphTypes.Erode, morphKernel);

            Mat bw = new Mat();
            Cv2.Threshold(grad, bw, 32, 255, ThresholdTypes.Binary | ThresholdTypes.Otsu);
            Cv2.BitwiseNot(bw, bw);

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

            var merged = MergeRects(rectangles, src.Width, src.Height);

            foreach (var rect in merged)
            {
                Cv2.Rectangle(debug, rect, Scalar.Red, 2);
            }

            List<Cv.Rect> filtered = new List<Cv.Rect>();
            for (int i = 0; i < merged.Count; i++)
            {
                var trimmed = TrimRect(thresh, merged[i]);
                if (trimmed.Width > 0)
                {
                    Mat roi = new Mat(thresh, trimmed);
                    if (HeuristicCheck(roi))
                    {
                        filtered.Add(merged[i]);
                    }
                }
            }

            //filtered = ZoneOfInterest(filtered, debug);

            foreach (var rect in filtered)
            {
                Cv2.Rectangle(debug, rect, Scalar.Orange, 2);
            }

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

                    groupRect.Inflate(5, 5);
                    groupRect = FixInvalidRects(groupRect, src.Width, src.Height);

                    Mat groupRoi = new Mat(src, groupRect);

                    byte[] buff;
                    if (Cv2.ImEncode(".tiff", groupRoi, out buff))
                    {
                        string output;
                        if(TryOcrAddress(buff, out output))
                        {
                            sb.AppendLine(output);
                            Cv2.Rectangle(debug, groupRect, Scalar.Purple, 6);
                        }
                        else
                        {
                            Cv2.Rectangle(debug, groupRect, Scalar.Lime, 2);
                        }
                        sb.AppendLine();
                    }
                }
            }

            return new OcrResult() { DebugImage = debug, OcrText = sb.ToString() };
        }

        private Cv.Rect TrimRect(Mat thresh, Cv.Rect rect)
        {
            Mat roi = new Mat(thresh, rect);
            Mat rowReduce = new Mat();
            Cv2.Reduce(roi, rowReduce, ReduceDimension.Row, ReduceTypes.Sum, MatType.CV_32S);

            var csv = rowReduce.Dump(DumpFormat.Csv);
            var values = csv.Replace("[", "").Replace("]", "").Split(',').ToList().ConvertAll(s => int.Parse(s) > 0 ? 1 : 0);

            int lCount = 0;
            int rCount = 0;
            foreach (var val in values)
            {
                if (val > 0)
                    break;
                lCount++;
            }

            values.Reverse();
            foreach (var val in values)
            {
                if (val > 0)
                    break;
                rCount++;
            }

            var x = rect.X + lCount;
            var width = Math.Max(0, rect.Width - (lCount + rCount));
            return new Cv.Rect(x, rect.Y, width, rect.Height);
        }

        private bool HeuristicCheck(Mat roi)
        {
            Mat rowReduce = new Mat();
            Mat colReduce = new Mat();
            Cv2.Reduce(roi, rowReduce, ReduceDimension.Row, ReduceTypes.Sum, MatType.CV_32S);
            Cv2.Reduce(roi, colReduce, ReduceDimension.Column, ReduceTypes.Sum, MatType.CV_32S);

            //var totPercent = Math.Round((double)Cv2.CountNonZero(roi) / (roi.Width * roi.Height), 2);

            /*Cv2.ImWrite("roi.tiff", roi);
            Cv2.ImWrite("roi-row.tiff", rowReduce);
            Cv2.ImWrite("roi-col.tiff", colReduce);*/

            if (!HeuristicColumnCheck(colReduce))
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

                        if (intersectArea > (lArea * 0.25) && intersectArea > (rArea * 0.25))
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

                    var hExpand = (int)Math.Round(width * 0.01, 0);
                    var l = new Cv.Rect(rects[i].X, rects[i].Y, rects[i].Width, rects[i].Height);
                    var r = new Cv.Rect(rects[j].X, rects[j].Y, rects[j].Width, rects[j].Height);
                    l.Inflate(hExpand / 2, 0);
                    r.Inflate(hExpand / 2, 0);

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

            var merged = new List<Cv.Rect>();
            for (int i = 0; i < rects.Count; i++)
            {
                rects[i] = FixInvalidRects(rects[i], width, height);

                if (!ignoreIndices.Contains(i))
                {
                    merged.Add(rects[i]);
                }
            }
            return merged;
        }

        private static Cv.Rect FixInvalidRects(Cv.Rect rect, int width, int height)
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

            return rect;
        }
    }
}
 