
HFONT fnt;
HDC hDC;
HGLRC hRC;
int iPixelFormat;
HWND hwnd;
int width=900,height=900;
float viewport_w=10,viewport_h=10;

unsigned int selbuf[512];
int viewport[4];

PIXELFORMATDESCRIPTOR pfd = { 0x28,   // size of this pfd
	1,                     // version number
	PFD_DRAW_TO_WINDOW |   // support window
	PFD_SUPPORT_OPENGL |
        PFD_DOUBLEBUFFER,   // support OpenGL,      // double buffered
	0,         // RGBA type
	24,                    // 24-bit color depth
	0, 0, 0, 0, 0, 0,      // color bits ignored
	0,                     // no alpha buffer
	0,                     // shift bit ignored
        0,                     // no accumulation buffer
	0, 0, 0, 0,            // accum bits ignored
	32,                    // 32-bit z-buffer
	0,                     // no stencil buffer
	0,                     // no auxiliary buffer
	0,                     // main layer
	0,                     // reserved
	0, 0, 0                // layer masks ignored
};

void init_opengl(){

  glEnable(GL_DEPTH_TEST);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(-viewport_w/2,viewport_w/2,-viewport_h/2,viewport_h/2,-100,10);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glViewport(0,0,width,height);

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);

  glEnable(GL_ALPHA);
  glGetIntegerv(GL_VIEWPORT,viewport);

  glClearColor(0,0,0,1);
  fnt=CreateFontW(40,0,0,0,200,0,0,0,RUSSIAN_CHARSET,OUT_DEFAULT_PRECIS,OUT_DEFAULT_PRECIS,CLEARTYPE_QUALITY,DEFAULT_PITCH|FF_DONTCARE,L"Century");

  if(!fnt)
    MessageBox(0,"nope","error creating font",0);
  else
    SelectObject(hDC, fnt);
  wglUseFontBitmapsW(hDC, 0, 2000, 1000); 
  glListBase(1000);

  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP);
  glTexEnvf(GL_TEXTURE_2D,GL_TEXTURE_ENV_MODE,GL_MODULATE);
 glEnable(GL_TEXTURE_2D);
 glTexImage2D(GL_TEXTURE_2D,0,GL_RGB,640,480,0,GL_RGB,GL_UNSIGNED_BYTE,texture);

}

void draw(){
 glColor3f(1,1,1);
 glBegin(GL_QUADS);
 glTexCoord2f(0,1);
 glVertex3f(-viewport_w/2,-viewport_w*3/8,0);
 glTexCoord2f(1,1);
 glVertex3f(viewport_w/2,-viewport_w*3/8,0);
 glTexCoord2f(1,0);
 glVertex3f(viewport_w/2,viewport_w*3/8,0);
 glTexCoord2f(0,0);
 glVertex3f(-viewport_w/2,viewport_w*3/8,0);
 glEnd();
}

LRESULT CALLBACK WinProc(HWND hwnd,UINT msg,WPARAM wParam,LPARAM lParam){
  switch(msg){
    case WM_CREATE:
    hDC=GetDC(hwnd);
    iPixelFormat = ChoosePixelFormat(hDC, &pfd);
    SetPixelFormat(hDC,iPixelFormat,&pfd);
    hRC = wglCreateContext( hDC );
    wglMakeCurrent(hDC, hRC);
    init_opengl();
    break;
    case WM_PAINT:
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      draw();
      glFinish();
      SwapBuffers(hDC);
    break;
    case WM_DESTROY:
      PostQuitMessage(0);
    break;
    default:
      return DefWindowProc(hwnd,msg,wParam,lParam);
  }
  return 0;
}

void createWindow(){
  MSG msg;
  char buf[256];
  WNDCLASSEXA wcex;

    wcex.cbSize = sizeof(WNDCLASSEX);
    wcex.style          = CS_HREDRAW | CS_VREDRAW;
    wcex.lpfnWndProc    = WinProc;
    wcex.cbClsExtra     = 0;
    wcex.cbWndExtra     = 0;
    wcex.hInstance      = 0;
    wcex.hIcon          = 0;
    wcex.hCursor        = 0;
    wcex.hbrBackground  = 0;
    wcex.lpszMenuName   = 0;
    wcex.lpszClassName  = "HELLO";
    wcex.hIconSm        =0;

  if(!RegisterClassExA(&wcex)){
    sprintf(buf,"RegisterClassEx failed with error %08X\n",GetLastError());
    MessageBox(0,buf,"Error",0);
    return;
  }

  hwnd = CreateWindowA("HELLO", "june 2022 game",WS_POPUP,
      500, 0, width, height, 0, 0,0, 0);

  if(!hwnd){ 
    sprintf(buf,"error:%08X\n",GetLastError());
    MessageBox(0,buf,"nope",0);
    return;
  }
  ShowWindow(hwnd,1);
  ShowCursor(0);
  UpdateWindow(hwnd);
  while(GetMessage(&msg,0,0,0)){
    TranslateMessage(&msg);
    DispatchMessage(&msg);
  }
}