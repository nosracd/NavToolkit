#!/usr/bin/env python3

import re
import sys
from glob import glob
from os.path import join
from typing import Any, NamedTuple

from cxxheaderparser import parser
from cxxheaderparser.errors import CxxParseError
from cxxheaderparser.parserstate import (
    ClassBlockState,
    NamespaceBlockState,
    NonClassBlockState,
    State,
)
from cxxheaderparser.types import (
    AnonymousName,
    Array,
    DecltypeSpecifier,
    EnumDecl,
    Field,
    Function,
    Method,
    Parameter,
    Pointer,
    Type,
    Typedef,
    Variable,
)
from cxxheaderparser.visitor import CxxVisitor
from inflection import camelize, underscore

NAMING_PATTERNS = {
    'lowercase': re.compile(r'^[a-z][a-z0-9]*$'),
    'UPPERCASE': re.compile(r'^[A-Z][A-Z0-9]*$'),
    'camelCase': re.compile(r'^[a-z][a-zA-Z0-9]*$'),
    'PascalCase': re.compile(r'\b[A-Z]\b|^[A-Z](?=.*[a-z])[a-zA-Z0-9]*$'),
    'snake_case': re.compile(r'^[a-z][a-z0-9_]*$'),
    'LOUD_SNAKE_CASE': re.compile(r'^[A-Z][A-Z0-9_]*$'),
}


def snake_case_but_allow_capital_C(word):
    # This is based on underscore from the inflection library.
    # MIT Licensed, (c) 2012-2015 Janne Vanhala
    word = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', word)
    word = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', word)
    word = word.replace('-', '_')
    return re.sub(
        r'[^C]|(?<!_)C(?!_)|(?<=_)C(?!$)(?!_)|(?<!_)(?<!^)C(?=_)',
        lambda m: m.group(0).lower(),
        word,
    )


RENAMERS = {
    'lowercase': lambda x: x.replace('_', '').lower(),
    'UPPERCASE': lambda x: x.replace('_', '').upper(),
    'camelCase': lambda x: camelize(x, False),
    'PascalCase': lambda x: camelize(x, True),
    'snake_case': snake_case_but_allow_capital_C,
    'LOUD_SNAKE_CASE': lambda x: underscore(x).upper(),
}


# Symbols to leave alone because I'm assuming they're mathy
MATH_BLACKLIST = set(
    """
    A a F MU x f expected_H expected_R
    PI pi tilt_P R0 P1700 DEG2RAD RAD2DEG pi_sq
""".split()
)

# Names of things marked const or constexpr that are specifically allowed to
# not be LOUD_SNAKE_CASE
CONST_BLACKLIST = {'forwarded_declval', 'value'}

# Matrix type variables have different naming rules
MATRIX_TYPES = {
    'Vector',
    'Matrix',
    'eig',
    'auto',
    'size_t',
    'Matrix3',
    'Vector3',
}


# What the state is and what typing pattern it should use
NAMING_RULE = {
    'FUNCTION_DECL': 'snake_case',
    'CXX_METHOD': 'snake_case',
    'NAMESPACE': 'snake_case',
    'CLASS_FIELD': 'snake_case',
    'FMT_FORMATTER': 'snake_case',
    'VAR_DECL': 'snake_case',
    'PARAMETER': 'snake_case',
    'CLASS_DECL': 'PascalCase',
    'TYPEDEF_DECL': 'PascalCase',
    'ENUM': 'PascalCase',
    'CONSTRUCTOR': 'PascalCase',
    'FORWARD_DECL': 'PascalCase',
    'DESTRUCTOR': 'PascalCase',
    'ENUM_VALUE': 'LOUD_SNAKE_CASE',
    'CONSTEXPR': 'LOUD_SNAKE_CASE',
    'CONST_VAR': 'LOUD_SNAKE_CASE',
    'EXTERN_NAMESPACE': 'LOUD_SNAKE_CASE',
    'STATIC_NAMESPACE': 'LOUD_SNAKE_CASE',
    'STATIC_CONST_CLASS': 'LOUD_SNAKE_CASE',
}

assert set(NAMING_RULE.values()) <= set(RENAMERS)
assert set(NAMING_RULE.values()) <= set(NAMING_PATTERNS)


# States to check if something looks like math for
CHECK_MATH_STATES = set(
    """
    FUNCTION_DECL CXX_METHOD CLASS_FIELD CONSTRUCTOR DESTRUCTOR PARAMETER
""".split()
)


# Store names that use the wrong rule and what they should be corrected to
class NameCorrection(NamedTuple):
    original_name: str
    corrected_name: str


# A visitor that the parser will use to check
# naming and keep locations
class LineNumberVisitor(CxxVisitor):
    def __init__(
        self, define_issues: dict[str, list[NameCorrection]], file: str
    ):
        self._ns_stack: list[str] = []
        self.file_name: str = file
        self.issues: dict[str, list[NameCorrection]] = define_issues

    # Some naming rules are different in namespace scopes.
    @property
    def in_namespace_scope(self) -> bool:
        return bool(self._ns_stack)

    @property
    def in_anonymous_namespace_scope(self) -> bool:
        return any(not ns_name for ns_name in self._ns_stack)

    def check_naming(
        self, state_type: str, name: str, location: str, format_str: str = ''
    ) -> None:
        if name.startswith('operator') or name == 'not_null':
            return
        if (
            self.in_namespace_scope or state_type in CHECK_MATH_STATES
        ) and self.looks_like_math(name, format_str):
            return
        rule: str = NAMING_RULE[state_type]
        renamer = RENAMERS[rule]
        new_name: str = renamer(name)
        # If in an anonymous namespace, allow loud snake case whenever
        # normal snake case is allowed to allow for file-level global types
        if (
            self.in_anonymous_namespace_scope
            and rule == 'snake_case'
            and name == RENAMERS['LOUD_SNAKE_CASE'](name)
        ):
            return
        # Allow UPPERCASE for Enum values
        if state_type == 'ENUM_VALUE' and name == RENAMERS['UPPERCASE'](name):
            return

        # There's an issue when renaming from LOUD_SNAKE_CASE to PascalCase it
        # becomes LOUDSNAKECASE instead of LoudSnakeCase, fix it here by
        # first converting to snake_case then camelizing or Pascalizing
        if name == RENAMERS['LOUD_SNAKE_CASE'](name) and name != RENAMERS[
            'UPPERCASE'
        ](name):
            snakey = snake_case_but_allow_capital_C(name)
            if rule == 'PascalCase':
                new_name = camelize(snakey, True)
            if rule == 'camelCase':
                new_name = camelize(snakey, False)

        # Store the old and revised version of name if
        # it doesn't meet naming standards
        if new_name != name:
            # It doesn't seem possible to get the line number of enum values,
            # so store them all as being on the line number of the enum decl,
            # but this means they will overwrite each other's keys so do this
            # to store all issues of one location in a list
            self.issues.setdefault(location, []).append(
                NameCorrection(original_name=name, corrected_name=new_name)
            )

    # Return the file and line number of a state
    def loc(self, state: State) -> str:
        return f'{self.file_name}:{state.location.lineno}'

    # Don't worry about naming for things that are mathy
    def looks_like_math(self, name: str, format_str: str) -> bool:
        tokens: list[str] = re.findall(r'[a-zA-Z0-9:_]+', format_str)
        # Filter using Matrix types and the xt:: prefix
        types: set = {
            t for t in tokens if t in MATRIX_TYPES or t.startswith('xt::')
        }
        if types and len(name) < 5:
            return True
        return name in MATH_BLACKLIST or name.startswith('C_') or '_C_' in name

    # Pointers are structured differently, check if it is one and
    # return the "Type" for all cases.
    def check_if_pointer(self, input_type) -> Type:
        if isinstance(input_type, Pointer):
            return input_type.ptr_to
        elif isinstance(input_type, Type):
            return input_type
        elif isinstance(input_type, Array):
            return input_type.array_of
        else:
            print(
                'Unexpected input_type in check_if_pointer: '
                f'{type(input_type).__name__}'
            )
            raise TypeError(
                'Unexpected input_type in check_if_pointer: '
                f'{type(input_type).__name__}'
            )

    # AnonymousName and DecltypeSpecifier can be returned by seg[-1]
    # but do not have many important attributes, check here
    # if it is one to avoid an attribute error
    def is_anonymous_name_or_decltype(self, seg: Any) -> bool:
        return isinstance(seg, (AnonymousName, DecltypeSpecifier))

    # Each of these on_ methods is called when the corresponding state
    # is encountered while the file is being parsed. Get the name of
    # each one and make sure their naming is correct.

    def on_namespace_start(self, state: NamespaceBlockState) -> None:
        ns_name: str = '::'.join(state.namespace.names or ())
        self._ns_stack.append(ns_name)
        if ns_name:
            self.check_naming('NAMESPACE', ns_name, self.loc(state))

    def on_namespace_end(self, state: NamespaceBlockState) -> None:
        self._ns_stack.pop()

    def on_function(self, state: NonClassBlockState, fn: Function) -> None:
        if self.is_anonymous_name_or_decltype(fn.name.segments[-1]):
            return
        # Don't worry about operators
        if isinstance(fn.return_type, Type):
            if not self.is_anonymous_name_or_decltype(
                fn.return_type.typename.segments[-1]
            ):
                if fn.return_type.typename.segments[-1].name == 'OPERATOR':
                    return
        name: str = fn.name.segments[-1].name
        location: str = self.loc(state)
        self.process_parameters(fn.parameters, location)
        self.check_naming('FUNCTION_DECL', name, location)

    def on_variable(self, state: State, v: Variable) -> None:
        if self.is_anonymous_name_or_decltype(v.name.segments[-1]):
            return
        # Don't worry about parameters of gtests, they're irregular
        if isinstance(v.type, Type):
            if not self.is_anonymous_name_or_decltype(
                v.type.typename.segments[0]
            ):
                const: bool = v.type.const
                function_name: str = v.type.typename.segments[0].name
                if 'TEST' in function_name:
                    return
        elif isinstance(v.type, Array):
            if not isinstance(v.type.array_of, Array):
                const: bool = v.type.array_of.const
        name: str = v.name.segments[-1].name
        location: str = self.loc(state)
        constexpr: bool = v.constexpr
        extern: bool = bool(v.extern)
        static: bool = v.static
        var_type: Type = self.check_if_pointer(v.type)
        format_str: str = var_type.format()

        if constexpr and name not in CONST_BLACKLIST:
            self.check_naming('CONSTEXPR', name, location, format_str)
        elif const and name not in CONST_BLACKLIST:
            self.check_naming('CONST_VAR', name, location, format_str)
        elif extern and self.in_namespace_scope:
            self.check_naming('EXTERN_NAMESPACE', name, location, format_str)
        elif static and self.in_namespace_scope:
            self.check_naming('STATIC_NAMESPACE', name, location, format_str)
        else:
            self.check_naming('VAR_DECL', name, location, format_str)

    def on_typedef(self, state: State, typedef: Typedef) -> None:
        name: str = typedef.name
        location: str = self.loc(state)
        self.check_naming('TYPEDEF_DECL', name, location)

    def on_enum(self, state: State, enum: EnumDecl) -> None:
        segs: list = enum.typename.segments
        name: str = segs[-1].name
        location: str = self.loc(state)
        for v in enum.values:
            self.check_naming('ENUM_VALUE', v.name, location)
        self.check_naming('ENUM', name, location)

    def on_class_start(self, state: ClassBlockState) -> None:
        segs: list = state.class_decl.typename.segments
        scope: str = segs[0].name
        name: str = segs[-1].name
        location: str = self.loc(state)

        # Special exception of normal naming rules
        if scope == 'fmt' and name == 'formatter':
            self.check_naming('FMT_FORMATTER', name, location)
        else:
            self.check_naming('CLASS_DECL', name, location)

    def on_class_method(self, state: ClassBlockState, method: Method) -> None:
        if self.is_anonymous_name_or_decltype(method.name.segments[-1]):
            return
        name: str = method.name.segments[-1].name
        location: str = self.loc(state)
        constructor: bool = method.constructor
        destructor: bool = method.destructor
        override: bool = method.override
        self.process_parameters(method.parameters, location)
        # Don't screw with overrides, since they have to match their base class
        if override:
            return
        elif constructor:
            self.check_naming('CONSTRUCTOR', name, location)
        elif destructor:
            self.check_naming('DESTRUCTOR', name, location)
        else:
            self.check_naming('CXX_METHOD', name, location)

    def on_class_field(self, state: ClassBlockState, f: Field) -> None:
        field_type: Type = self.check_if_pointer(f.type)
        constexpr: bool = f.constexpr
        const: bool = field_type.const
        static: bool = f.static
        name: str = f.name
        location: str = self.loc(state)
        format_str: str = field_type.format()

        if (constexpr or const) and static and name not in CONST_BLACKLIST:
            self.check_naming('STATIC_CONST_CLASS', name, location, format_str)
        else:
            self.check_naming('CLASS_FIELD', name, location, format_str)

    def on_forward_decl(self, state, fdecl):
        name: str = fdecl.typename.segments[0].name
        location: str = self.loc(state)
        self.check_naming('FORWARD_DECL', name, location)

    def process_parameters(self, parameters: list[Parameter], location: str):
        for param in parameters:
            if param.name:
                self.check_naming(
                    'PARAMETER', param.name, location, param.format()
                )


# cxxheaderparser can't handle lines with #define so this function handles
# them and makes sure they are LOUD_SNAKE_CASE
def parse_defines(filename: str) -> dict[str, list[NameCorrection]]:
    issues: dict[str, list[NameCorrection]] = {}
    with open(filename) as f:
        for line_number, line in enumerate(f, start=1):
            # Not robust enough comment check but should handle
            # anything following code formatting standards in this project
            if '#define' in line and not line.startswith('//'):
                define: str = line.split()[1]
                # Can also have functions in #define statements, do this to
                # just get the name of the function
                if '(' in define:
                    define = define.split('(', 1)[0]
                if define != RENAMERS['LOUD_SNAKE_CASE'](define):
                    issues.setdefault(filename + f':{line_number}', []).append(
                        NameCorrection(
                            original_name=define,
                            corrected_name=RENAMERS['LOUD_SNAKE_CASE'](define),
                        )
                    )
    return issues


# Parse a file and return naming issues
def parse_file(filename: str) -> dict[str, list[NameCorrection]]:
    with open(filename) as fh:
        content: str = fh.read()
    content = preprocess_file(content)
    # Find any issues with #define statements naming and pass
    # them here so they are included with the other issues.
    visitor = LineNumberVisitor(
        define_issues=parse_defines(filename), file=filename
    )
    file_parser = parser.CxxParser(filename, content, visitor)
    file_parser.parse()

    return visitor.issues


def comment_out_defines(code: str) -> str:
    """
    Comments out all #define statements in a C++ code string,
    including multi-line defines that use backslashes
    """
    pattern = re.compile(r'[ \t]*#define(?:[^\n]*\\[ \t]*\n)*[^\n]+')

    def add_comments(match: re.Match) -> str:
        block = match.group(0)
        # Prefix every line in the block with //
        return re.sub(r'^', '//', block, flags=re.MULTILINE)

    return pattern.sub(add_comments, code)


# Matches the opening of any #if block: #ifdef, #ifndef, #if
IFDEF_RE = re.compile(r'[ \t]*#[ \t]*(?:ifdef|ifndef|if)\b')
# Matches #endif
ENDIF_RE = re.compile(r'[ \t]*#[ \t]*endif\b')


def comment_out_ifdefs(code: str) -> str:
    """
    Comments out every #ifdef / #ifndef / #if ... #endif block
    """
    lines: list[str] = code.splitlines(keepends=True)
    result: list[str] = []
    depth: int = 0

    for line in lines:
        is_open = bool(IFDEF_RE.match(line))
        is_close = bool(ENDIF_RE.match(line))

        if is_open:
            depth += 1

        if depth > 0:
            nl = '\n' if line.endswith('\n') else ''
            result.append('//' + line.rstrip('\n') + nl)
        else:
            result.append(line)

        if is_close:
            depth -= 1

    return ''.join(result)


CXX_OFF_RE = re.compile(r'[ \t]*//[ \t]*cxxheaderparser off')
CXX_ON_RE = re.compile(r'[ \t]*//[ \t]*cxxheaderparser on')


def comment_out_cxx_header_parser_off_sections(code: str) -> str:
    """
    Comment out everything between //cxxheaderparser off and
    //cxxheaderparser on
    Use this to make naming checks skip some blocks
    of code that are causing issues.
    """
    lines: list[str] = code.splitlines(keepends=True)
    result: list[str] = []
    inside: bool = False

    for line in lines:
        if CXX_OFF_RE.match(line):
            inside = True
            result.append(line)
        elif CXX_ON_RE.match(line):
            inside = False
            result.append(line)
        elif inside:
            result.append('//' + line)
        else:
            result.append(line)

    return ''.join(result)


def fix_gtests(input: str) -> str:
    """
    gtests cause parse errors because of their abnormal formatting.
    Fix them here to turn them into methods while still keeping their names.
    Not all of the test types are used but fix them all here for
    possible future use.
    """

    # First fix tests that end in , ) which can't be parsed
    # ie: INSTANTIATE_TYPED_TEST_SUITE_P(EwC_SLOW,
    # FusionStrategyTests, FusionStrategyTestsTypes, );
    input = input.replace(', );', ');')

    TEST_MACRO_MAP = {
        'TEST': ';void test',
        'TEST_F': ';void test_f',
        'TEST_P': ';void test_p',
        'TYPED_TEST': ';void typed_test',
        'TYPED_TEST_SUITE': ';void typed_test_suite',
        'TYPED_TEST_P': ';void typed_test_p',
        'TYPED_TEST_SUITE_P': ';void typed_test_suite_p',
        'REGISTER_TYPED_TEST_SUITE_P': ';void register_typed_test_suite_p',
        'INSTANTIATE_TYPED_TEST_SUITE_P': ';void '
        'instantiate_typed_test_suite_p',
        'INSTANTIATE_TEST_SUITE_P': ';instantiate_test_suite_p',
        'ERROR_MODE_SENSITIVE_TEST': ';void error_mode_sensitive_test',
    }
    pattern = rf'\b({"|".join(re.escape(k) for k in TEST_MACRO_MAP.keys())})\('

    def replace_match(match):
        macro_name = match.group(1)
        return f'{TEST_MACRO_MAP[macro_name]}('

    # Run the substitution in a single pass
    return re.sub(pattern, replace_match, input)


def preprocess_file(input: str) -> str:
    """
    Remove problematic blocks from files that are to be parsed
    """
    # SFINAE have too many irregularities, comment them out
    input = re.sub(
        r'^(.*?\b\w*SFINAE\w*\b\s*\(.*)$', r'// \1', input, flags=re.MULTILINE
    )
    # Replace TEST in files using gtest with a void method
    input = fix_gtests(input)
    # Comment out stuff in `/cxxheaderparser off` blocks
    input = comment_out_cxx_header_parser_off_sections(input)
    # Remove #define statements, preprocessor doesn't actually fix
    input = comment_out_defines(input)
    # Remove #ifdef stuff
    input = comment_out_ifdefs(input)

    return input


def build_filenames() -> list[str]:
    """
    Provides filenames to parse.
    """
    filenames: list[str] = [
        path
        for srcdir in ('test', 'src/navtk', 'examples')
        for ext in ('*.cpp', '*.hpp')
        for path in glob(join(srcdir, '**', ext), recursive=True)
    ]
    return filenames


def main():
    filenames: list[str] = build_filenames()
    corrections: dict[str, list[NameCorrection]] = {}
    failed_a_parse: bool = False

    for file in filenames:
        try:
            issues: dict[str, list[NameCorrection]] = parse_file(file)
            corrections = corrections | issues
        except CxxParseError as e:
            print(e)
            failed_a_parse = True
    for location, old_and_fixed_names in corrections.items():
        # Does not print naming errors for failed parses.
        # Check if it's a list for enums with multiple
        # enum values with incorrect naming rules
        for names in old_and_fixed_names:
            print(
                f'At {location}, {names.original_name}'
                f' should be {names.corrected_name}'
            )
    if corrections:
        print('Naming issues detected; check failed.')
        sys.exit(1)
    elif failed_a_parse:
        print('Parse error detected; check failed.')
        sys.exit(1)
    else:
        print('No naming issues detected; check successful.')
        sys.exit(0)


if __name__ == '__main__':
    main()
