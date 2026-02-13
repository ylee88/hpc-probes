module foo_mod

  implicit none

  integer, parameter :: dp = kind(1.0d0)

  !$acc routine(foo) vector
  !$acc routine(foo_ijk)

contains

  subroutine foo(data, n, inner_iters, ni, nj, nk, nn)
    implicit none
    integer, intent(in) :: n, inner_iters, ni, nj, nk, nn
    real(dp), intent(inout) :: data(ni, nj, nk, nn)

    integer :: i, j, k, r
    real(dp) :: x, a, b, c

    a = 1.0000001_dp
    b = 0.9999993_dp
    c = 0.1234567_dp

    !$acc loop vector collapse(3) private(i,j,k,x,r)
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

  end subroutine foo

  subroutine foo_ijk(data, i, j, k, n, inner_iters, ni, nj, nk, nn)
    implicit none
    integer, intent(in) :: i, j, k, n
    integer, intent(in) :: inner_iters, ni, nj, nk, nn
    real(dp), intent(inout) :: data(ni, nj, nk, nn)

    integer :: r
    real(dp) :: x, a, b, c

    a = 1.0000001_dp
    b = 0.9999993_dp
    c = 0.1234567_dp

    x = data(i,j,k,n)
    do r = 1, inner_iters
      x = a*x + c
      x = x - b*x*x
      x = x + 0.000001_dp * (a - b)
    end do
    data(i,j,k,n) = x

  end subroutine foo_ijk

end module foo_mod
