import sys
import ezdxf
import os
from ezdxf import recover

folder = r'C:\Users\jc2713\OneDrive - Imperial College London (1)\Projects\Japan2024_May_KSAV\Japan_visit\Sample_design\AlexFab'
file = r'Hall-FMR_v2.1-20211117_original.dxf'
path = os.path.join(folder, file)


try:  # low level structure repair:
    doc, auditor = recover.readfile(path)
except IOError:
    print(f"Not a DXF file or a generic I/O error.")
    sys.exit(1)
except ezdxf.DXFStructureError:
    print(f"Invalid or corrupted DXF file: {path}.")
    sys.exit(2)

# DXF file can still have unrecoverable errors, but this is maybe
# just a problem when saving the recovered DXF file.
if auditor.has_errors:
    print(f"Found unrecoverable errors in DXF file: {path}.")
    auditor.print_error_report()