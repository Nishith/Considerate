package edu.cmu.nishith;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.LinkedList;
import java.util.List;
import java.util.Timer;
import java.util.TimerTask;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.highgui.VideoCapture;
import org.opencv.objdetect.CascadeClassifier;

import android.content.Context;
import android.graphics.Bitmap;
import android.media.AudioManager;
import android.media.Ringtone;
import android.media.RingtoneManager;
import android.net.Uri;
import android.util.Log;
import android.view.SurfaceHolder;

class FdView extends SampleCvViewBase {
	private static final String TAG = "FdView";
	private Mat                 mRgba;
	private Mat                 mGray;

	private int                 filter = 5;
	private int 				tick   = 0;
	Timer revertTimer = new Timer();
	int orignalRingMode                = 0;
	boolean ringer = false;

	Context local_context;
	FdActivity act = new FdActivity();

	class revertRinger extends TimerTask {
		@Override
		public void run() {
			AudioManager ringer = (AudioManager) local_context.getSystemService(Context.AUDIO_SERVICE);
			ringer.setRingerMode(AudioManager.RINGER_MODE_NORMAL);
		}	
	}

	private CascadeClassifier   mCascade;

	public FdView(Context context) {
		super(context);

		local_context = context;
		revertTimer = new Timer();

		try {
			// Get the cascade file provided by OpenCV and load it into the system.
			InputStream is = context.getResources().openRawResource(R.raw.lbpcascade_frontalface);
			File cascadeDir = context.getDir("cascade", Context.MODE_PRIVATE);
			File cascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
			FileOutputStream os = new FileOutputStream(cascadeFile);

			byte[] buffer = new byte[4096];
			int bytesRead;
			while ((bytesRead = is.read(buffer)) != -1) {
				os.write(buffer, 0, bytesRead);
			}
			is.close();
			os.close();
			
			//Get the cascade classifier object which will be used later for object recognition. 
			mCascade = new CascadeClassifier(cascadeFile.getAbsolutePath());
			if (mCascade.empty()) {
				Log.e(TAG, "Failed to load cascade classifier");
				mCascade = null;
			} else
				Log.i(TAG, "Loaded cascade classifier from " + cascadeFile.getAbsolutePath());

			cascadeFile.delete();
			cascadeDir.delete();

		} catch (IOException e) {
			e.printStackTrace();
			Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
		}
	}

	@Override
	public void surfaceChanged(SurfaceHolder _holder, int format, int width, int height) {
		super.surfaceChanged(_holder, format, width, height);

		synchronized (this) {
			// initialize Mats before usage
			mGray = new Mat();
			mRgba = new Mat();
		}
	}

	@Override
	protected Bitmap processFrame(VideoCapture capture) {
		// The detection works on a gray scale image. So, we extract a gray scale image for
		// object detection and an RGBA image which will be displayed back to the user with 
		// the object inside a green square.
		capture.retrieve(mRgba, Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
		capture.retrieve(mGray, Highgui.CV_CAP_ANDROID_GREY_FRAME);

		if (mCascade != null) {
			int height = mGray.rows();
			int faceSize = Math.round(height * FdActivity.minFaceSize);
			List<Rect> faces = new LinkedList<Rect>();
			// Send the image to detectMultiScale which will do the detection
			// The image co-ordinates where the object was detected will be filled in the 
			// faces array.
			mCascade.detectMultiScale(mGray, faces, 1.1, 2, 2
					, new Size(faceSize, faceSize));

			// If the object was detected for lesser number of frames than our threshold and
			// then disappeared, we clear the tick counter. This helps remove a lot of false 
			// positives.
			if(faces.isEmpty() && tick > 0) {
				tick = 0;
			}
			
			//For each detection we increment the tick counter
			if (!faces.isEmpty()) {
				tick++;
				// If the tick counter is greater than the threshold we switch the phone to vibrate
				if (tick > filter) {
					tick = 0;
					Uri notification = RingtoneManager.getDefaultUri(RingtoneManager.TYPE_NOTIFICATION);
					Ringtone r = RingtoneManager.getRingtone(local_context, notification);
					r.play();
					AudioManager ringer = (AudioManager) local_context.getSystemService(Context.AUDIO_SERVICE);
					orignalRingMode = ringer.getRingerMode();
					ringer.setRingerMode(AudioManager.RINGER_MODE_VIBRATE);
					// Start a timer to revert the ringer back to the normal mode.
					if(this.ringer == false){
						this.ringer = true;
						revertTimer = new Timer();
						TimerTask revert = new revertRinger();
						revertTimer.schedule(revert, 10000);
					}
				}
			}
			// Draw the rectangles around the objects detected.
			for (Rect r : faces)
				Core.rectangle(mRgba, r.tl(), r.br(), new Scalar(0, 255, 0, 255), 3);
		}
		
		Bitmap bmp = Bitmap.createBitmap(mRgba.cols(), mRgba.rows(), Bitmap.Config.ARGB_8888);

		if (Utils.matToBitmap(mRgba, bmp))
			return bmp;

		bmp.recycle();
		return null;
	}

	@Override
	public void run() {
		super.run();

		synchronized (this) {
			// Explicitly deallocate Mats
			if (mRgba != null)
				mRgba.release();
			if (mGray != null)
				mGray.release();
			
			mRgba = null;
			mGray = null;
		}
	}
}

