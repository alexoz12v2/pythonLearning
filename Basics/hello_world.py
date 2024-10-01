# the print method is the hello world to write to stdout
print("Hello World")

msg0 = 'first string'
num = 4

# f-strings to insert variables inside them
msgNum = f'the number with f-strings is {num}'
print(msgNum);

# .format() method to insert varaibles too
# specify a label, followed by ":" its format, eg 2f, meaning 2
#   decimal places
# placeholder can also be numbered or empty, not necessarily named
txt = 'the number with .format() is {price:.2f}'.format(price = 49)
print(txt)

# - repetition opereator (*), 
# - concatenation operator (+),
# - .title() or .capitalize() method to capitalize
# - .replace(old, new) replace substring
newMsg = msg0.title() * 3 + msg0.replace('first', 'another');
print(newMsg)
print(f'msg0.isalpha(): {msg0.isalpha()}')

withSpaces = 'ends with spaces     ';
print(withSpaces)
print('now without spaces: ' + withSpaces.rstrip())

# Note: the == operator has higher precedence than the 
# in operator hence this returns false
print(f'is there a in apple? {"a" in "apple" == True}')
print(f'is there a in apple for real? {("a" in "apple") == True}')

# note: when you print a list you cannot concatenate with a string...
print('default split separator is space'.split())

# but JS style works
print(list('arrayofletterswithlist'), '\n')

# list comprehension
print([char for char in 'letters'], '\n')

# explained later
import sys

float_info = sys.float_info
print('Information about the internal representation of numbers\n'
  + 'in your machine with sys.float_info:\n',
  f"  max         : {float_info.max} # Maximum representable positive finite float\n",
  f"  max_exp     : {float_info.max_exp} # Maximum exponent\n",
  f"  max_10_exp  : {float_info.max_10_exp} # Maximum base-10 exponent\n",
  f"  min         : {float_info.min} # Minimum positive normalized float\n",
  f"  min_exp     : {float_info.min_exp} # Minimum exponent\n",
  f"  min_10_exp  : {float_info.min_10_exp} # Minimum base-10 exponent\n",
  f"  dig         : {float_info.dig} # Maximum decimal digits\n",
  f"  mant_dig    : {float_info.mant_dig} # Number of bits in the mantissa\n",
  f"  epsilon     : {float_info.epsilon} # Difference between 1 and the least value greater than 1\n",
  f"  radix       : {float_info.radix} # Base of the floating-point representation\n",
  f"  rounds      : {float_info.rounds} # Rounding mode")

# number declarations
print('Some number examples: \n',
      f'\t{-43242}\n',
      f'\t{3.15}\n',
      f'\t{1+3.23j}\n')

# separators for readability
big = 14_000_000

# multiple assignment
x, y, z = 1+1j, 2.23-34j, 3
err = '(complex not supported)';

print('Numerical Operations:\n\t'
  f'x = {x}, y = {y}\n\t',
  f'int(5321412342314312).bit_length() = {int(5321412342314312).bit_length()}\n\t',
  f'x + y            = {x + y}\n\t',
  f'x - y            = {x - y}\n\t',
  f'x * y            = {x * y}\n\t',
  f'x / y            = {x / y}\n\t',
  f'x.real // y.imag = {x.real // y.imag} {err}\n\t',
  f'x.real % y.imag  = {x.real % y.imag} {err}\n\t',
  f'-x               = {-x}\n\t',
  f'+x               = {+x}\n\t',
  f'abs(x)           = {abs(x)} (computes the magnitude for complex)\n\t',
  f'int("10")        = {int("10")}\n\t',
  f'complex(x.real, y.imag) = {complex(x.real, y.imag)}\n\t',
  f'y.conjugate()      = {y.conjugate()}\n\t',
  f'divmod(x.real, 30) = {divmod(x.real, 30)}\n\t',
  f'pow(x, y)          = {pow(x, y)}\n\t',
  f'x ** y             = {x ** y}\n\t',
  'To take the phase of a complex, import cmath and use cmath.phase'
)

# calling a method on a lietral requires parenthesis or constructor
print(f'\n\n\n(-2.0).is_integer() = {(-2.0).is_integer()}')

# trying out iterators
lst = [1, 2, 3, 4, 4]
print(f'initial list of len(lst) = {len(lst)}:\n\t{lst}')

lst.append('another type')
lst.insert(2, 'fsd')
print(f'after append(.) and insert(2, .): \n\t{lst}')

del lst[4]
popped = lst.pop() # can pass an index too
print(f'after del lst[4] and pop(): \n\t{lst}, popped: {popped}')

lst.remove('fsd')
print(f'after lst.remove(\'fsd\'): \n\t{lst}')

# lst.insert(3, 'fst') sorting between str and int -> TypeError
lst.sort() # version which returns a non mutable -> sorted(lst)
print(f'after lst.sort():\n\t{lst}')

lst.reverse()
print(f'after lst.reverse()\n\t{lst}')

print(f'negative index index: go backwards, lst[-1]: {lst[-1]}')

#trying out iterators
print('iterating through lst with iterators')
for num in lst: # remember: indentation and colon
  print(f'\tcurrent element: {num}')


def ordinalStr(num):
  match num:
    case 1:
      return '1st'
    case 2:
      return '2nd'
    case 3:
      return '3rd'
    case _:
      return str(num) + 'th'

# trying out iterators for myself
class Thing:
  # private method starts with __

  # private variable
  _numbers = [ordinalStr(x) for x in range(1, 10)]

  # constructor
  def __init__(self, x):
    # another variable
    self._firstStr = self._numbers[0]
    self._arg = x

  # iterable interface (__.*__ is a built-in function)
  def __iter__(self):
    return self._numbers.__iter__()

  # to print it directly
  def __repr__(self):
    className = type(self).__name__
    return f'{className}()'

  def getArg(self):
    return self._arg;

thing = Thing(32)
print(f'custom object thing: {thing}')

for n in thing:
  print(f'\tcurrent number string: {n}')

print(f'strings are also sequences: "gg" in "eggs":'
  f' {"gg" in "eggs"}')

# list of lists
mdLst = [[] for i in range(3)]
mdLst[0].append(2)
mdLst[1].append(2.32)
mdLst[2].append(2 + 7j)
print(f'multidimensional list: {mdLst}')
