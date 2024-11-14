dir = '/spo82/ana/240521_simu/240807/rsfish/image/';
res_dir = "/spo82/ana/240521_simu/240807/rsfish/';
timeFile = res_dir + "times.txt";
startTime = getTime();
names = newArray(19); // Specifying size during initialization
for (i = 1; i < 19; i++) {
    // suffix = IJ.pad(i, 2); // Ensure the number is two digits
    names[i] = "no" + i + ".tif";
}

for (i=1; i<19; i++) {

	open(dir + names[i]);

	run("RS-FISH", "image=" + names[i] + " " + 
	        "mode=Advanced anisotropy=1.0000 robust_fitting=RANSAC spot_intensity=[Linear Interpolation] add image_min=53 image_max=239 " + 
            "sigma=2.178 threshold=0.005 support=3 min_inlier_ratio=0.12 max_error=0.99 spot_intensity_threshold=15.03 background=[No background subtraction] background_subtraction_max_error=0.05 background_subtraction_min_inlier_ratio=0.12 "+
            "results_file=[" +  res_dir + names[i] + ".csv] ");
    /*       
	run("RS-FISH", "image=" + names[i] + " " + 
			"mode=Advanced anisotropy=01 robust_fitting=[RANSAC] use_anisotropy " + 
			"image_min=53 image_max=239 sigma=1.5 threshold=0.005 support=3 " +
			"min_inlier_ratio=0.163 max_error=0.9041912 spot_intensity_threshold=0 " +
			"background=[No background subtraction] " +
			" results_file=[" +  res_dir + names[i] + ".csv] ");
	*/
	close();

}
exeTime = getTime() - startTime; //in miliseconds
File.append(exeTime + "\n ", timeFile);
