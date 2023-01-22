from tensorflow import cast, reduce_sum, int32, divide, subtract, square, reduce_mean

def R_squared(y, prediction):
  prediction = cast(prediction, dtype=int32)

  unexplained_error = reduce_sum(square(subtract(y, prediction)))
  total_error = reduce_sum(square(subtract(y, reduce_mean(y))))
  return subtract(1, divide(unexplained_error, total_error))