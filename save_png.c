void save_png(const char *filename, unsigned char *limage, int width, int height) {

 FILE *fp = fopen(filename, "wb");
 if (!fp) {
  fprintf(stderr, "Could not open file %s for writing\n", filename);
  return;
 }

 png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
 png_infop info_ptr = png_create_info_struct(png_ptr);

 if(setjmp(png_jmpbuf(png_ptr))) {
  png_destroy_write_struct(&png_ptr, &info_ptr);
  fclose(fp);
  printf("Error\n");
  return;
 }

 png_init_io(png_ptr, fp);
 png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,PNG_COMPRESSION_TYPE_DEFAULT,PNG_FILTER_TYPE_DEFAULT);
 png_write_info(png_ptr, info_ptr);

 if (!limage) {
  fprintf(stderr, "limage buffer is NULL\n");
  return;
 }
 unsigned char *row=malloc(width*10);
 for (int y = 0; y < height; y++) {
  png_write_row(png_ptr, limage + y * width * 3);
 }
 png_write_end(png_ptr, NULL);
 png_destroy_write_struct(&png_ptr, &info_ptr);
 fclose(fp);
 printf("PNG saved successfully to %s\n", filename);
}
