program main

  use omp_lib, only: omp_get_wtime
  use foo_mod, only: foo, foo_ijk, dp

  implicit none

  integer, parameter :: ni=256, nj=256, nk=32, nn=32
  integer, parameter :: steps=10
  integer, parameter :: inner_iters=10

  real(dp), allocatable :: data(:,:,:,:)
  real(dp) :: t0, t1, cpu_s, gpu_s, checksum_cpu, checksum_gpu

  real(dp) :: x, a, b, c

  integer :: i, j, k, n, s, r

  allocate(data(ni,nj,nk,nn))

  do n=1,nn
    do k=1,nk
      do j=1,nj
        do i=1,ni
          data(i,j,k,n) = 0.001_dp*i + 0.002_dp*j + 0.003_dp*k + 0.01_dp*n
        end do
      end do
    end do
  end do

  ! CPU baseline
  t0 = omp_get_wtime()
  do s = 1, steps
    do n = 1, nn
      call foo(data, n, inner_iters, ni, nj, nk, nn)
    end do
  end do
  t1 = omp_get_wtime()
  cpu_s = t1 - t0

  checksum_cpu = 0.0_dp
  do n=1,nn
    checksum_cpu = checksum_cpu + sum(data(:,:,:,n))
  end do

  print *, "----------------------------------------"
  print *, "GPU offload **with** foo() function call"
  print *, "----------------------------------------"

  ! reset
  do n=1,nn
    do k=1,nk
      do j=1,nj
        do i=1,ni
          data(i,j,k,n) = 0.001_dp*i + 0.002_dp*j + 0.003_dp*k + 0.01_dp*n
        end do
      end do
    end do
  end do

  ! GPU offload
  !$omp target data map(tofrom:data)

  ! warm-up
  !$omp target teams loop bind(teams)
  do n = 1, nn
    call foo(data, n, inner_iters, ni, nj, nk, nn)
  end do

  t0 = omp_get_wtime()
  do s = 1, steps
    !$omp target teams loop bind(teams)
    do n = 1, nn
      call foo(data, n, inner_iters, ni, nj, nk, nn)
    end do
  end do
  t1 = omp_get_wtime()
  gpu_s = t1 - t0

  !$omp end target data

  checksum_gpu = 0.0_dp
  do n=1,nn
    checksum_gpu = checksum_gpu + sum(data(:,:,:,n))
  end do

  print *, "CPU time (s):", cpu_s
  print *, "GPU time (s):", gpu_s
  print *, "Speedup     :", cpu_s / gpu_s
  print *, "Abs diff    :", abs(checksum_cpu - checksum_gpu)

  if (abs(checksum_cpu - checksum_gpu) > 1.0e-6_dp * max(1.0_dp, abs(checksum_cpu))) then
    error stop "FAILED: checksum mismatch"
  end if
  print *, "PASSED"


  print *, "-------------------------------------------"
  print *, "GPU offload **without** foo() function call"
  print *, "-------------------------------------------"

  ! reset
  do n=1,nn
    do k=1,nk
      do j=1,nj
        do i=1,ni
          data(i,j,k,n) = 0.001_dp*i + 0.002_dp*j + 0.003_dp*k + 0.01_dp*n
        end do
      end do
    end do
  end do

  !$omp target data map(tofrom:data)
  a = 1.0000001_dp
  b = 0.9999993_dp
  c = 0.1234567_dp


  t0 = omp_get_wtime()
  do s = 1, steps
    !$omp target teams distribute parallel do collapse(4) private(i,j,k,n,x,r)
    do n = 1, nn
      do k = 1, nk
        do j = 1, nj
          do i = 1, ni
            x = data(i,j,k,n)
            do r = 1, inner_iters
              x = a*x + c
              x = x - b*x*x
              x = x + 0.000001_dp * (a - b)
            end do
            data(i,j,k,n) = x
          end do
        end do
      end do
    end do
    !
  end do
  t1 = omp_get_wtime()
  gpu_s = t1 - t0

  !$omp end target data

  checksum_gpu = 0.0_dp
  do n=1,nn
    checksum_gpu = checksum_gpu + sum(data(:,:,:,n))
  end do

  print *, "CPU time (s):", cpu_s
  print *, "GPU time (s):", gpu_s
  print *, "Speedup     :", cpu_s / gpu_s
  print *, "Abs diff    :", abs(checksum_cpu - checksum_gpu)

  if (abs(checksum_cpu - checksum_gpu) > 1.0e-6_dp * max(1.0_dp, abs(checksum_cpu))) then
    error stop "FAILED: checksum mismatch"
  end if
  print *, "PASSED"


  print *, "----------------------------------------------------"
  print *, "GPU offload **functor-like** foo_ijk() function call"
  print *, "----------------------------------------------------"

  ! reset
  do n=1,nn
    do k=1,nk
      do j=1,nj
        do i=1,ni
          data(i,j,k,n) = 0.001_dp*i + 0.002_dp*j + 0.003_dp*k + 0.01_dp*n
        end do
      end do
    end do
  end do

  !$omp target data map(tofrom:data)

  t0 = omp_get_wtime()
  do s = 1, steps
    !$omp target teams distribute parallel do collapse(4) private(i,j,k,n,x,r)
    do n = 1, nn
      do k = 1, nk
        do j = 1, nj
          do i = 1, ni
            call foo_ijk(data, i, j, k, n, inner_iters, ni, nj, nk, nn)
          end do
        end do
      end do
    end do
    !
  end do
  t1 = omp_get_wtime()
  gpu_s = t1 - t0

  !$omp end target data

  checksum_gpu = 0.0_dp
  do n=1,nn
    checksum_gpu = checksum_gpu + sum(data(:,:,:,n))
  end do

  print *, "CPU time (s):", cpu_s
  print *, "GPU time (s):", gpu_s
  print *, "Speedup     :", cpu_s / gpu_s
  print *, "Abs diff    :", abs(checksum_cpu - checksum_gpu)

  if (abs(checksum_cpu - checksum_gpu) > 1.0e-6_dp * max(1.0_dp, abs(checksum_cpu))) then
    error stop "FAILED: checksum mismatch"
  end if
  print *, "PASSED"


end program main
