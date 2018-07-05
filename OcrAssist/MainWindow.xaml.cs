using Microsoft.Win32;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

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
                Mat src = new Mat(ofd.FileName, ImreadModes.GrayScale);

                /*Mat scaled = src.Clone();
                Cv2.PyrDown(scaled, scaled);
                Cv2.PyrUp(scaled, scaled);

                Mat grad = new Mat();
                var morphKernel = Cv2.GetStructuringElement(MorphShapes.Ellipse, new OpenCvSharp.Size(3, 3));
                Cv2.MorphologyEx(scaled, grad, MorphTypes.Gradient, morphKernel);

                Mat bw = new Mat();
                Cv2.Threshold(grad, bw, 32, 255, ThresholdTypes.Binary | ThresholdTypes.Otsu);

                Mat connected = new Mat();
                morphKernel = Cv2.GetStructuringElement(MorphShapes.Rect, new OpenCvSharp.Size(8, 1));
                Cv2.MorphologyEx(bw, connected, MorphTypes.Close, morphKernel);*/

                Mat mask = Mat.Zeros(src.Size(), MatType.CV_8UC1);
                var mserExtractor = MSER.Create();

                OpenCvSharp.Point[][] msers;
                OpenCvSharp.Rect[] bboxes;
                mserExtractor.DetectRegions(src, out msers, out bboxes);

                for (int i = 0; i < bboxes.Length; i++)
                {
                    //Cv2.Rectangle(clone, bboxes[i], Scalar.Black, 15);

                    Mat roi = new Mat(mask, bboxes[i]);
                    roi.SetTo(Scalar.White);
                }

                Mat grad = new Mat();
                Mat kernel = new Mat(1, 50, MatType.CV_8UC1, Scalar.White);
                Cv2.MorphologyEx(mask, grad, MorphTypes.Dilate, kernel);

                OpenCvSharp.Point[][] contours;
                HierarchyIndex[] hierarchy;
                Cv2.FindContours(grad, out contours, out hierarchy, RetrievalModes.External, ContourApproximationModes.ApproxNone);

                for (int i = 0; i < hierarchy.Length; i++)
                {
                    Cv2.DrawContours(src, contours, i, Scalar.Black, 4);
                }

                //

                MemoryStream ms = new MemoryStream();
                grad.WriteToStream(ms);
                var imageSource = new BitmapImage();
                imageSource.BeginInit();
                imageSource.StreamSource = ms;
                imageSource.EndInit();

                Image.Source = imageSource;

                MemoryStream ms2 = new MemoryStream();
                src.WriteToStream(ms2);
                var imageSource2 = new BitmapImage();
                imageSource2.BeginInit();
                imageSource2.StreamSource = ms2;
                imageSource2.EndInit();

                Original.Source = imageSource2;
            }
        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            Test();
        }
    }
}
