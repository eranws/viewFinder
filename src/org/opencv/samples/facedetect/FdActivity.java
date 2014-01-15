package org.opencv.samples.facedetect;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.objdetect.CascadeClassifier;

import android.app.Activity;
import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.WindowManager;
import android.widget.ExpandableListView.ExpandableListContextMenuInfo;

public class FdActivity extends Activity implements CvCameraViewListener2 {

    private static final String    TAG                 = "OCVSample::Activity";
    private static final Scalar    FACE_RECT_COLOR     = new Scalar(0, 255, 0, 255);
    public static final int        JAVA_DETECTOR       = 0;
    public static final int        NATIVE_DETECTOR     = 1;

    private static final Scalar    
    COLOR_BLACK     = new Scalar(0, 0, 0, 255),
    COLOR_RED = new Scalar(255, 0, 0, 128),
    COLOR_BLUE = new Scalar(0, 0, 255, 128);

    private MenuItem               mItemFace50;
    private MenuItem               mItemFace40;
    private MenuItem               mItemFace30;
    private MenuItem               mItemFace20;
    private MenuItem               mItemType;

    private Mat                    mRgba;
    private Mat                    mGray;
    private File                   mCascadeFile;
    private CascadeClassifier      mJavaDetector;
    private DetectionBasedTracker  mNativeDetector;

    private int                    mDetectorType       = NATIVE_DETECTOR;
    private String[]               mDetectorName;

    private float                  mRelativeFaceSize   = 0.2f;
    private int                    mAbsoluteFaceSize   = 0;

    private CameraBridgeViewBase   mOpenCvCameraView;

    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");

                    // Load native library after(!) OpenCV initialization
                    System.loadLibrary("detection_based_tracker");

                    try {
                        // load cascade file from application resources
                        InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
                        File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                        mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
                        FileOutputStream os = new FileOutputStream(mCascadeFile);

                        byte[] buffer = new byte[4096];
                        int bytesRead;
                        while ((bytesRead = is.read(buffer)) != -1) {
                            os.write(buffer, 0, bytesRead);
                        }
                        is.close();
                        os.close();

                        mJavaDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
                        if (mJavaDetector.empty()) {
                            Log.e(TAG, "Failed to load cascade classifier");
                            mJavaDetector = null;
                        } else
                            Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());

                        mNativeDetector = new DetectionBasedTracker(mCascadeFile.getAbsolutePath(), 0);

                        cascadeDir.delete();

                    } catch (IOException e) {
                        e.printStackTrace();
                        Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
                    }

                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public FdActivity() {
        mDetectorName = new String[2];
        mDetectorName[JAVA_DETECTOR] = "Java";
        mDetectorName[NATIVE_DETECTOR] = "Native (tracking)";

        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.face_detect_surface_view);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.fd_activity_surface_view);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.enableFpsMeter();
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this, mLoaderCallback);
    }

    public void onDestroy() {
        super.onDestroy();
        mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();
    }

    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {

        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();
        
        
        Mat can = mGray.clone();
		Imgproc.Canny(mGray, can, 80, 90);
		
		int threshold = 80;
		double minLineLength=150;
		double maxLineGap=0 ;
				
		Mat lines = new Mat();
		Imgproc.HoughLinesP(can, lines, 3, Math.PI / 60, threshold, minLineLength, maxLineGap);
	    //Imgproc.cvtColor(can, mRgba, Imgproc.COLOR_GRAY2BGRA, 4);
		can.release();
		
		 for (int x = 0; x < lines.cols(); x++) 
		    {
		          double[] vec = lines.get(0, x);
		          double x1 = vec[0], 
		                 y1 = vec[1],
		                 x2 = vec[2],
		                 y2 = vec[3];
		          Point start = new Point(x1, y1);
		          Point end = new Point(x2, y2);
		          Core.line(mRgba, start, end, new Scalar(255,0,0), 3);
		    }
		 

        if (mAbsoluteFaceSize == 0) {
            int height = mGray.rows();
            if (Math.round(height * mRelativeFaceSize) > 0) {
                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
            }
            mNativeDetector.setMinFaceSize(mAbsoluteFaceSize);
        }

        MatOfRect faces = new MatOfRect();

        if (mDetectorType == JAVA_DETECTOR) {
            if (mJavaDetector != null)
                mJavaDetector.detectMultiScale(mGray, faces, 1.1, 2, 2, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
                        new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
        }
        else if (mDetectorType == NATIVE_DETECTOR) {
            if (mNativeDetector != null)
                mNativeDetector.detect(mGray, faces);
        }
        else {
            Log.e(TAG, "Detection method is not selected!");
        }

        Rect[] facesArray = faces.toArray();
        for (int i = 0; i < Math.min(1,facesArray.length); i++)
        {
            //Core.rectangle(mRgba, facesArray[i].tl(), facesArray[i].br(), FACE_RECT_COLOR, 3);
            
        	
            Point center = new Point(facesArray[i].x + facesArray[i].width / 2,
            		facesArray[i].y + facesArray[i].height / 2);
            int radius = facesArray[i].width / 2;
            Scalar color = FACE_RECT_COLOR;
            Core.circle(mRgba, center, radius, color);   
            
            float ratio = 3f;
            Rect expandedFace = new Rect(
            		(int)(facesArray[i].x - (ratio-1)/2*facesArray[i].width),
            		(int)(facesArray[i].y - (ratio-1)/2*facesArray[i].height),
            		(int)(ratio * facesArray[i].width),
            		(int)(ratio * facesArray[i].height));
            		
            Core.rectangle(mRgba, expandedFace.tl(), expandedFace.br(),
            		COLOR_BLUE, 3);
            
            for (int j = 0; j < lines.cols(); j++) 
		    {
		          double[] vec = lines.get(0, j);
		          double x1 = vec[0], 
		                 y1 = vec[1],
		                 x2 = vec[2],
		                 y2 = vec[3];
		          Point start = new Point(x1, y1);
		          Point end = new Point(x2, y2);
		          
		          if (start.inside(expandedFace) || end.inside(expandedFace))
		          {
		        	  Core.line(mRgba, start, end, new Scalar(255,255,0), 8);
		          }
		    }
		 
            
            
            
            //width crop
            double posX = center.x / mRgba.width();
            Point pw1 = new Point(0, 0);
            Point pw2 = new Point(mRgba.width(), mRgba.height());
            
            //Log.i(TAG, ""+posX);
            if (posX < 1.0/3)
            {
            	pw1.x = 3 * center.x;
            }
            else if (posX < 1.0/2)
            {
            	pw2.x = (3 * center.x - mRgba.width()) / 2;
            }
            else if (posX < 2.0/3)
            {
             	pw1.x = 3.0 / 2 * center.x;
            }
            else
            {
            	pw2.x = (3 * center.x - 2 * mRgba.width());
            }
            
            //height crop
            double posY = center.y / mRgba.height();
            Point ph1 = new Point(0, 0);
            Point ph2 = new Point(mRgba.width(), mRgba.height());
            
            //Log.i(TAG, ""+posX);
            if (posY < 1.0/3)
            {
            	ph1.y = 3 * center.y;
            }
            else if (posY < 1.0/2)
            {
            	ph2.y = (3 * center.y - mRgba.height()) / 2;
            }
            else if (posY < 2.0/3)
            {
             	ph1.y = 3.0 / 2 * center.y;
            }
            else
            {
            	ph2.y = (3 * center.y - 2 * mRgba.height());
            }

            //Core.rectangle(mRgba, pw1, pw2, COLOR_RED, Core.FILLED);
            //Core.rectangle(mRgba, ph1, ph2, COLOR_BLUE, Core.FILLED);
            
            Scalar s = new Scalar(0.8);
            
            Mat rgbaInnerWindow = mRgba.submat((int)ph1.y, (int)ph2.y, (int)ph1.x, (int)ph2.x);
            Core.multiply(rgbaInnerWindow, s, rgbaInnerWindow);
            Mat rgbaInnerWindow2 = mRgba.submat((int)pw1.y, (int)pw2.y, (int)pw1.x, (int)pw2.x);
            Core.multiply(rgbaInnerWindow2, s, rgbaInnerWindow2);
            

//            Mat invertcolormatrix= new Mat(rgbaInnerWindow.rows(),rgbaInnerWindow.cols(), rgbaInnerWindow.type(), new Scalar(255,255,255));
 //           Core.mul(invertcolormatrix, rgbaInnerWindow, rgbaInnerWindow);
            
//            Imgproc.cvtColor(rgbaInnerWindow, rgbaInnerWindow, Imgproc.COLOR_GRAY2BGRA, 4);
            
            
            
            
            
                        
        }
       
        return mRgba;
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
        mItemFace50 = menu.add("Face size 50%");
        mItemFace40 = menu.add("Face size 40%");
        mItemFace30 = menu.add("Face size 30%");
        mItemFace20 = menu.add("Face size 20%");
        mItemType   = menu.add(mDetectorName[mDetectorType]);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);
        if (item == mItemFace50)
            setMinFaceSize(0.5f);
        else if (item == mItemFace40)
            setMinFaceSize(0.4f);
        else if (item == mItemFace30)
            setMinFaceSize(0.3f);
        else if (item == mItemFace20)
            setMinFaceSize(0.2f);
        else if (item == mItemType) {
            int tmpDetectorType = (mDetectorType + 1) % mDetectorName.length;
            item.setTitle(mDetectorName[tmpDetectorType]);
            setDetectorType(tmpDetectorType);
        }
        return true;
    }

    private void setMinFaceSize(float faceSize) {
        mRelativeFaceSize = faceSize;
        mAbsoluteFaceSize = 0;
    }

    private void setDetectorType(int type) {
        if (mDetectorType != type) {
            mDetectorType = type;

            if (type == NATIVE_DETECTOR) {
                Log.i(TAG, "Detection Based Tracker enabled");
                mNativeDetector.start();
            } else {
                Log.i(TAG, "Cascade detector enabled");
                mNativeDetector.stop();
            }
        }
    }
}
