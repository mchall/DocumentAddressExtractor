using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OcrAssist
{
    public class OcrResult
    {
        public string OcrText { get; set; }
        public Mat DebugImage { get; set; }
    }
}