%string = type { ptr, i32, i32 }

@dnl = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1
@snl = private unnamed_addr constant [6 x i8] c"%.*s\0A\00", align 1
@d = private unnamed_addr constant [3 x i8] c"%d\00", align 1
@msn = private unnamed_addr constant [6 x i8] c"%ms%n\00", align 1
@error.1 = private unnamed_addr constant [15 x i8] c"runtime error\0A\00", align 1

declare i32 @printf(ptr, ...)

declare i32 @scanf(ptr, ...)

declare void @exit(i32)

declare noalias ptr @malloc(i32)

define void @printInt(i32 %0) {
entry:
  %call = call i32 (ptr, ...) @printf(ptr @dnl, i32 %0)
  ret void
}

define void @printString(ptr %0) {
entry:
  %buf_ptr = getelementptr inbounds %string, ptr %0, i32 0, i32 0
  %len_ptr = getelementptr inbounds %string, ptr %0, i32 0, i32 1
  %buf = load ptr, ptr %buf_ptr, align 8
  %len = load i32, ptr %len_ptr, align 4
  %call = call i32 (ptr, ...) @printf(ptr @snl, i32 %len, ptr %buf)
  ret void
}

define i32 @readInt() {
entry:
  %buf = alloca i32, align 4
  %call = call i32 (ptr, ...) @scanf(ptr @d, ptr %buf)
  %buf1 = load i32, ptr %buf, align 4
  ret i32 %buf1
}

define ptr @readString() {
entry:
  %buf = alloca ptr, align 8
  %len = alloca i32, align 4
  %call = call i32 (ptr, ...) @scanf(ptr @msn, ptr %buf, ptr %len)
  %buf1 = load ptr, ptr %buf, align 8
  %len2 = load i32, ptr %len, align 4
  %string = tail call ptr @malloc(i32 ptrtoint (ptr getelementptr (%string, ptr null, i32 1) to i32))
  %buf_ptr = getelementptr inbounds %string, ptr %string, i32 0, i32 0
  %len_ptr = getelementptr inbounds %string, ptr %string, i32 0, i32 1
  %len_ptr3 = getelementptr inbounds %string, ptr %string, i32 0, i32 2
  store ptr %buf1, ptr %buf_ptr, align 8
  store i32 %len2, ptr %len_ptr, align 4
  store i32 %len2, ptr %len_ptr3, align 4
  ret ptr %string
}

define void @error() {
entry:
  %call = call i32 (ptr, ...) @printf(ptr @error.1)
  call void @exit(i32 1)
  ret void
}