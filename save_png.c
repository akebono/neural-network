void save_png(const char *filename, unsigned char *image, int width, int height) {
 FILE *fp = fopen(filename, "wb");
 if (!fp) {
  fprintf(stderr, "Could not open file %s for writing\n", filename);
  return;
 }
 png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
 if (!png) {
  fclose(fp);
  return;
 }
 png_infop info = png_create_info_struct(png);
 if (!info) {
  png_destroy_write_struct(&png, NULL);
  fclose(fp);
  return;
 }
 // Set up error handling
 if (setjmp(png_jmpbuf(png))) {
  png_destroy_write_struct(&png, &info);
  fclose(fp);
  return;
 }
 png_init_io(png, fp);

 // Set image attributes
 png_set_IHDR(png, info, width, height, 8, 
   PNG_COLOR_TYPE_RGB,           // RGB format
   PNG_INTERLACE_NONE,
   PNG_COMPRESSION_TYPE_DEFAULT,
   PNG_FILTER_TYPE_DEFAULT);

 png_write_info(png, info);

 // Write image data
 unsigned char ** row;
 row=malloc(height*sizeof(char*));
 for(int y=0;y<height;y++){
  row[y] = malloc(width * 3); // RGB has 3 channels
  memcpy(row[y],image+y*3*width,3*width);
 }
 unsigned char * row1 = malloc(width * 3); // RGB has 3 channels.

 for (int y = 0; y < height; y++) {

  for (int x = 0; x < width; x++) {
   row[y][x * 3] = (png_byte)(x % 256);        // Red
   row[y][x * 3 + 1] = (png_byte)(y % 256);    // Green
   row[y][x * 3 + 2] = (png_byte)((x + y) % 256); // Blue
  }

  png_write_row(png, row[y]);
 }
 png_write_end(png, NULL);
 // Cleanup
 png_destroy_write_struct(&png, &info);
 fclose(fp);
 printf("PNG saved successfully to %s\n", filename);
}
