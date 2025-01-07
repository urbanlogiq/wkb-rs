// Copyright (c) 2022 UrbanLogiq
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#![deny(clippy::unused_async)]
#![deny(clippy::unnecessary_wraps)]
#![deny(clippy::redundant_closure_for_method_calls)]
#![deny(clippy::cloned_instead_of_copied)]
#![deny(clippy::needless_pass_by_value)]
#![deny(clippy::match_wildcard_for_single_variants)]
#![deny(clippy::single_match_else)]
#![deny(clippy::if_not_else)]
#![deny(clippy::cast_lossless)]
#![deny(clippy::explicit_iter_loop)]
#![deny(clippy::semicolon_if_nothing_returned)]
#![deny(clippy::map_flatten)]
#![deny(clippy::default_trait_access)]
#![feature(trait_alias)]

/// The `wkb-rs` library implements parsing of WKB (Well Known Binary) geometry
/// data into `geo-types` structures and serialization of `geo-types` geometry
/// into WKB data.
///
/// At present only 2D types and little-endian byte formats are supported.
use geo_types::geometry::{
    Coord, GeometryCollection, LineString, MultiLineString, MultiPoint, MultiPolygon, Point,
    Polygon,
};
use geo_types::Geometry;
use num_derive::FromPrimitive;
use num_traits::Float;
use num_traits::FromPrimitive;
use std::convert::TryFrom;
use std::error::Error;
use std::fmt;
use std::fmt::Debug;
use std::io::Cursor;
use std::io::Read;

pub trait NumTy = Float + Debug + From<f64> + Into<f64>;

#[derive(Debug)]
pub enum WkbError {
    IoError(std::io::Error),
    UnsupportedType,
    UnsupportedEndianess,
    InconsistentType,
}

impl fmt::Display for WkbError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Error for WkbError {}

impl From<std::io::Error> for WkbError {
    fn from(err: std::io::Error) -> Self {
        Self::IoError(err)
    }
}

#[derive(FromPrimitive)]
enum Endian {
    Big = 0,
    Little = 1,
}

#[derive(FromPrimitive)]
enum GeomTy {
    Point = 1,
    LineString = 2,
    Polygon = 3,
    MultiPoint = 4,
    MultiLineString = 5,
    MultiPolygon = 6,
    GeometryCollection = 7,
}

struct LittleEndianReader<'a, 'b> {
    cursor: &'b mut Cursor<&'a [u8]>,
}

impl LittleEndianReader<'_, '_> {
    #[inline(always)]
    fn read_u32(&mut self) -> Result<u32, std::io::Error> {
        let mut r = [0u8; 4];
        self.cursor.read_exact(&mut r)?;

        Ok(u32::from_le_bytes(r))
    }

    #[inline(always)]
    fn read_f64(&mut self) -> Result<f64, std::io::Error> {
        let mut r = [0u8; 8];
        self.cursor.read_exact(&mut r)?;

        Ok(f64::from_le_bytes(r))
    }
}

/// The Wkb type is a thin wrapper over an owned buffer containing WKB data
/// encoded as bytes.
#[derive(Clone)]
pub struct Wkb(Vec<u8>);

impl Wkb {
    /// Constructs a new Wkb structure from an existing Vec<u8>.
    ///
    /// # Examples
    ///
    /// ```
    /// use geo_types::Geometry;
    /// use wkb_rs::Wkb;
    ///
    /// let point_wkb = vec![
    ///    0x1, 0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xf0, 0x3f, 0x0, 0x0, 0x0, 0x0, 0x0,
    ///    0x0, 0x0, 0x40,
    /// ];
    /// let wkb = Wkb::new(point_wkb);
    /// let point: Geometry<f64> = wkb.try_into().unwrap();
    /// match point {
    ///     Geometry::Point(p) => {
    ///         let point: geo_types::Point = (1.0, 2.0).into();
    ///         assert_eq!(point, p);
    ///     }
    ///     _ => panic!()
    /// }
    /// ```
    pub fn new(wkb: Vec<u8>) -> Self {
        Self(wkb)
    }

    /// Take ownership of the constructed WKB data.
    ///
    /// # Examples
    ///
    /// ```
    /// use geo_types::{Geometry, Point};
    /// use wkb_rs::Wkb;
    ///
    /// let point: Point = (1.0, 2.0).into();
    /// let wkb: Wkb = Geometry::Point(point).try_into().unwrap();
    /// let data = wkb.take();
    /// let point_wkb = vec![
    ///    0x1, 0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xf0, 0x3f, 0x0, 0x0, 0x0, 0x0, 0x0,
    ///    0x0, 0x0, 0x40,
    /// ];
    /// assert_eq!(data, point_wkb);
    /// ```
    pub fn take(self) -> Vec<u8> {
        self.0
    }

    /// Parse a Geometry type from a slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use geo_types::Geometry;
    /// use wkb_rs::Wkb;
    ///
    /// const POINT_WKB: &[u8] = &[
    ///    0x1, 0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xf0, 0x3f, 0x0, 0x0, 0x0, 0x0, 0x0,
    ///    0x0, 0x0, 0x40,
    /// ];
    ///
    /// let geo: Geometry<f64> = Wkb::from_slice(POINT_WKB).unwrap();
    /// ```
    pub fn from_slice<T: NumTy>(data: &[u8]) -> Result<Geometry<T>, WkbError> {
        let mut cursor = Cursor::new(data);
        read_wkb(&mut cursor)
    }
}

#[inline]
fn write_u32(buffer: &mut Vec<u8>, value: u32) {
    buffer.extend(value.to_le_bytes());
}

#[inline]
fn write_value<T: NumTy>(buffer: &mut Vec<u8>, value: T) {
    let value: f64 = value.into();
    buffer.extend(value.to_le_bytes());
}

#[inline]
fn write_ring<T: NumTy>(buffer: &mut Vec<u8>, ring: &LineString<T>) {
    let points = &ring.0;
    write_u32(buffer, points.len() as u32);
    for point in points {
        write_value(buffer, point.x);
        write_value(buffer, point.y);
    }
}

#[inline]
fn write_point<T: NumTy>(buffer: &mut Vec<u8>, p: &Point<T>) {
    buffer.push(Endian::Little as u8);
    write_u32(buffer, GeomTy::Point as u32);
    write_value(buffer, p.x());
    write_value(buffer, p.y());
}

#[inline]
fn write_line_string<T: NumTy>(buffer: &mut Vec<u8>, l: &LineString<T>) {
    buffer.push(Endian::Little as u8);

    let len = l.0.len();
    write_u32(buffer, GeomTy::LineString as u32);
    write_u32(buffer, len as u32);

    for point in &l.0 {
        write_value(buffer, point.x);
        write_value(buffer, point.y);
    }
}

#[inline]
fn write_polygon<T: NumTy>(buffer: &mut Vec<u8>, p: &Polygon<T>) {
    buffer.push(Endian::Little as u8);

    write_u32(buffer, GeomTy::Polygon as u32);
    let exterior = p.exterior();
    let interiors = p.interiors();
    let num_rings = (interiors.len() + 1) as u32;
    write_u32(buffer, num_rings);
    write_ring(buffer, exterior);

    for ring in interiors {
        write_ring(buffer, ring);
    }
}

fn write_geometry<T: NumTy>(buffer: &mut Vec<u8>, geom: &Geometry<T>) {
    match geom {
        Geometry::Point(p) => {
            write_point(buffer, p);
        }
        Geometry::LineString(l) => {
            write_line_string(buffer, l);
        }
        Geometry::Polygon(p) => {
            write_polygon(buffer, p);
        }
        Geometry::MultiPoint(mp) => {
            buffer.push(Endian::Little as u8);

            write_u32(buffer, GeomTy::MultiPoint as u32);
            write_u32(buffer, mp.0.len() as u32);

            mp.iter().for_each(|p| write_point(buffer, p));
        }
        Geometry::MultiLineString(ml) => {
            buffer.push(Endian::Little as u8);

            write_u32(buffer, GeomTy::MultiLineString as u32);
            write_u32(buffer, ml.0.len() as u32);

            ml.iter().for_each(|l| write_line_string(buffer, l));
        }
        Geometry::MultiPolygon(mp) => {
            buffer.push(Endian::Little as u8);

            write_u32(buffer, GeomTy::MultiPolygon as u32);
            write_u32(buffer, mp.0.len() as u32);

            mp.iter().for_each(|p| write_polygon(buffer, p));
        }
        Geometry::GeometryCollection(gc) => {
            buffer.push(Endian::Little as u8);

            write_u32(buffer, GeomTy::GeometryCollection as u32);
            write_u32(buffer, gc.len() as u32);

            gc.iter().for_each(|g| write_geometry(buffer, g));
        }
        _ => unimplemented!(),
    }
}

impl<T: NumTy> TryFrom<&Geometry<T>> for Wkb {
    type Error = WkbError;

    /// # Examples
    ///
    /// ```
    /// use geo_types::{Geometry, Point};
    /// use wkb_rs::Wkb;
    ///
    /// let point: Point = (1.0, 2.0).into();
    /// let wkb = Wkb::try_from(&Geometry::Point(point)).unwrap();
    /// let data = wkb.take();
    /// let point_wkb = vec![
    ///    0x1, 0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xf0, 0x3f, 0x0, 0x0, 0x0, 0x0, 0x0,
    ///    0x0, 0x0, 0x40,
    /// ];
    /// assert_eq!(data, point_wkb);
    /// ```
    fn try_from(geom: &Geometry<T>) -> Result<Wkb, Self::Error> {
        let mut buffer = Vec::new();

        write_geometry(&mut buffer, geom);

        Ok(Self(buffer))
    }
}

impl<T: NumTy> TryFrom<Geometry<T>> for Wkb {
    type Error = WkbError;

    /// # Examples
    ///
    /// ```
    /// use geo_types::{Geometry, Point};
    /// use wkb_rs::Wkb;
    ///
    /// let point: Point = (1.0, 2.0).into();
    /// let wkb = Wkb::try_from(Geometry::Point(point)).unwrap();
    /// let data = wkb.take();
    /// let point_wkb = vec![
    ///    0x1, 0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xf0, 0x3f, 0x0, 0x0, 0x0, 0x0, 0x0,
    ///    0x0, 0x0, 0x40,
    /// ];
    /// assert_eq!(data, point_wkb);
    /// ```
    fn try_from(geom: Geometry<T>) -> Result<Wkb, Self::Error> {
        Self::try_from(&geom)
    }
}

fn read_coordinate<T: NumTy>(reader: &mut LittleEndianReader) -> Result<Coord<T>, WkbError> {
    let x: T = reader.read_f64()?.into();
    let y: T = reader.read_f64()?.into();

    Ok(Coord { x, y })
}

fn read_coordinates<T: NumTy>(reader: &mut LittleEndianReader) -> Result<Vec<Coord<T>>, WkbError> {
    let num_points = reader.read_u32()? as usize;
    let mut points = Vec::with_capacity(num_points);

    for _ in 0..num_points {
        points.push(read_coordinate(reader)?);
    }

    Ok(points)
}

fn read_wkb<T: NumTy>(cursor: &mut Cursor<&[u8]>) -> Result<Geometry<T>, WkbError> {
    let mut endianness = [0u8; 1];
    cursor.read_exact(&mut endianness)?;

    let endianness = Endian::from_u8(endianness[0]).ok_or(WkbError::UnsupportedEndianess)?;
    let mut reader = match endianness {
        Endian::Big => return Err(WkbError::UnsupportedEndianess),
        Endian::Little => LittleEndianReader { cursor },
    };

    let ty_value = reader.read_u32()?;

    let ty_flags = ty_value >> 29;
    let masked_ty_value = ty_value & ((1 << 29) - 1);

    let (has_z, has_m, has_srid) = (
        (ty_flags & 0b100) != 0,
        (ty_flags & 0b010) != 0,
        (ty_flags & 0b001) != 0,
    );

    if has_z || has_m {
        return Err(WkbError::UnsupportedType);
    }

    let ty = GeomTy::from_u32(masked_ty_value).ok_or(WkbError::UnsupportedType)?;

    if has_srid {
        _ = reader.read_u32()?;
    }

    match ty {
        GeomTy::Point => read_coordinate(&mut reader).map(|p| Geometry::Point(Point::from(p))),
        GeomTy::LineString => {
            read_coordinates(&mut reader).map(|p| Geometry::LineString(LineString::from(p)))
        }
        GeomTy::Polygon => {
            let num_rings = reader.read_u32()? as usize;
            let mut rings = (0..num_rings)
                .map(|_| Ok(LineString::new(read_coordinates::<T>(&mut reader)?)))
                .collect::<Result<Vec<_>, WkbError>>()?;
            let exterior = rings.swap_remove(0);

            Ok(Geometry::Polygon(Polygon::new(exterior, rings)))
        }
        GeomTy::MultiPoint => {
            let num_points = reader.read_u32()? as usize;
            let points = (0..num_points)
                .map(|_| match read_wkb::<T>(cursor)? {
                    Geometry::Point(l) => Ok(l),
                    _ => Err(WkbError::InconsistentType),
                })
                .collect::<Result<Vec<_>, WkbError>>()?;
            Ok(Geometry::MultiPoint(MultiPoint::new(points)))
        }
        GeomTy::MultiLineString => {
            let num_linestrings = reader.read_u32()? as usize;
            let linestrings = (0..num_linestrings)
                .map(|_| match read_wkb::<T>(cursor)? {
                    Geometry::LineString(l) => Ok(l),
                    _ => Err(WkbError::InconsistentType),
                })
                .collect::<Result<Vec<_>, WkbError>>()?;
            Ok(Geometry::MultiLineString(MultiLineString::new(linestrings)))
        }
        GeomTy::MultiPolygon => {
            let num_polygons = reader.read_u32()? as usize;
            let polygons = (0..num_polygons)
                .map(|_| match read_wkb::<T>(cursor)? {
                    Geometry::Polygon(l) => Ok(l),
                    _ => Err(WkbError::InconsistentType),
                })
                .collect::<Result<Vec<_>, WkbError>>()?;
            Ok(Geometry::MultiPolygon(MultiPolygon::new(polygons)))
        }
        GeomTy::GeometryCollection => {
            let num_geometries = reader.read_u32()? as usize;
            let geometries = (0..num_geometries)
                .map(|_| read_wkb::<T>(cursor))
                .collect::<Result<Vec<_>, WkbError>>()?;
            Ok(Geometry::GeometryCollection(GeometryCollection::new_from(
                geometries,
            )))
        }
    }
}

impl<T: NumTy> TryFrom<Wkb> for Geometry<T> {
    type Error = WkbError;

    /// # Examples
    ///
    /// ```
    /// use geo_types::Geometry;
    /// use wkb_rs::Wkb;
    ///
    /// let point_wkb = vec![
    ///    0x1, 0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xf0, 0x3f, 0x0, 0x0, 0x0, 0x0, 0x0,
    ///    0x0, 0x0, 0x40,
    /// ];
    /// let wkb = Wkb::new(point_wkb);
    /// let point: Geometry<f64> = Geometry::<f64>::try_from(wkb).unwrap();
    /// match point {
    ///     Geometry::Point(p) => {
    ///         let point: geo_types::Point = (1.0, 2.0).into();
    ///         assert_eq!(point, p);
    ///     }
    ///     _ => panic!()
    /// }
    /// ```
    fn try_from(geom: Wkb) -> Result<Geometry<T>, Self::Error> {
        let mut cursor = Cursor::new(geom.0.as_slice());
        read_wkb(&mut cursor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BE_POINT_WKB: &[u8] = &[
        0x0, 0x0, 0x0, 0x0, 0x1, 0x3f, 0xf0, 0x0, 0x0, 0x0, 0x0, 0x00, 0x00, 0x40, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0,
    ];

    const POINT_WKB_SRID: &[u8] = &[
        0x1, 0x1, 0x0, 0x0, 0x20, 0xe6, 0x10, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xf0, 0x3f,
        0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x40,
    ];

    const POINT_WKB: &[u8] = &[
        0x1, 0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xf0, 0x3f, 0x0, 0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x40,
    ];
    const POINT: &[f64] = &[1.0, 2.0];

    const LINESTRING_WKB: &[u8] = &[
        0x1, 0x2, 0x0, 0x0, 0x0, 0x2, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xf0, 0x3f, 0x0,
        0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x40, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x40, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x8, 0x40,
    ];
    const LINESTRING: &[&[f64]] = &[&[1.0, 2.0], &[2.0, 3.0]];

    const POLYGON_WKB: &[u8] = &[
        0x1, 0x3, 0x0, 0x0, 0x0, 0x1, 0x0, 0x0, 0x0, 0x5, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xf0,
        0x3f, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xf0, 0x3f,
        0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xf0, 0x3f, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0, 0xf0, 0x3f, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0,
    ];
    const POLYGON: &[&[&[f64]]] = &[&[
        &[0.0, 0.0],
        &[1.0, 0.0],
        &[1.0, 1.0],
        &[0.0, 1.0],
        &[0.0, 0.0],
    ]];

    const POLYGON_WITH_HOLE_WKB: &[u8] = &[
        0x1, 0x3, 0x0, 0x0, 0x0, 0x2, 0x0, 0x0, 0x0, 0x5, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x24,
        0x40, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x24, 0x40,
        0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x24, 0x40, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0, 0x24, 0x40, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0, 0x5, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xf0, 0x3f, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0, 0xf0, 0x3f, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xf0, 0x3f, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0, 0x40, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x40, 0x0, 0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x40, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x40, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
        0xf0, 0x3f, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xf0, 0x3f, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xf0,
        0x3f,
    ];
    const POLYGON_WITH_HOLE: &[&[&[f64]]] = &[
        &[
            &[0.0, 0.0],
            &[10.0, 0.0],
            &[10.0, 10.0],
            &[0.0, 10.0],
            &[0.0, 0.0],
        ],
        &[
            &[1.0, 1.0],
            &[1.0, 2.0],
            &[2.0, 2.0],
            &[2.0, 1.0],
            &[1.0, 1.0],
        ],
    ];

    const MULTIPOINT_WKB: &[u8] = &[
        0x1, 0x4, 0x0, 0x0, 0x0, 0x2, 0x0, 0x0, 0x0, 0x1, 0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0xf0, 0x3f, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x40, 0x1, 0x1, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xf0, 0x3f, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x40,
    ];
    const MULTIPOINT: &[&[f64]] = &[&[1.0, 2.0], &[1.0, 2.0]];

    const MULTILINESTRING_WKB: &[u8] = &[
        0x1, 0x5, 0x0, 0x0, 0x0, 0x2, 0x0, 0x0, 0x0, 0x1, 0x2, 0x0, 0x0, 0x0, 0x2, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xf0, 0x3f, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x40, 0x0,
        0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x40, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x8, 0x40, 0x1, 0x2, 0x0,
        0x0, 0x0, 0x2, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x24, 0x40, 0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x24, 0x40, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x26, 0x40, 0x0, 0x0, 0x0, 0x0, 0x0,
        0x0, 0x39, 0x40,
    ];
    const MULTILINESTRING: &[&[&[f64]]] =
        &[&[&[1.0, 2.0], &[2.0, 3.0]], &[&[10.0, 10.0], &[11.0, 25.0]]];

    const MULTIPOLYGON_WKB: &[u8] = &[
        0x1, 0x6, 0x0, 0x0, 0x0, 0x2, 0x0, 0x0, 0x0, 0x1, 0x3, 0x0, 0x0, 0x0, 0x1, 0x0, 0x0, 0x0,
        0x5, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x24, 0x40, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x24, 0x40, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x24, 0x40, 0x0,
        0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x24, 0x40, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x1, 0x3, 0x0, 0x0, 0x0,
        0x1, 0x0, 0x0, 0x0, 0x5, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xf0, 0x3f, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0, 0xf0, 0x3f, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xf0, 0x3f, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0, 0x40, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x40, 0x0, 0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x40, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x40, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
        0xf0, 0x3f, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xf0, 0x3f, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xf0,
        0x3f,
    ];
    const MULTIPOLYGON: &[&[&[&[f64]]]] = &[
        &[&[
            &[0.0, 0.0],
            &[10.0, 0.0],
            &[10.0, 10.0],
            &[0.0, 10.0],
            &[0.0, 0.0],
        ]],
        &[&[
            &[1.0, 1.0],
            &[1.0, 2.0],
            &[2.0, 2.0],
            &[2.0, 1.0],
            &[1.0, 1.0],
        ]],
    ];

    const GEOMETRYCOLLECTION_WKB: &[u8] = &[
        0x1, 0x7, 0x0, 0x0, 0x0, 0x2, 0x0, 0x0, 0x0, 0x1, 0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x40, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x1, 0x3, 0x0, 0x0, 0x0, 0x1,
        0x0, 0x0, 0x0, 0x5, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xf0, 0x3f, 0x0, 0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xf0, 0x3f, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
        0xf0, 0x3f, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xf0,
        0x3f, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    ];
    const GEOMETRYCOLLECTION_POINT: &[f64] = &[2.0, 0.0];
    const GEOMETRYCOLLECTION_POLYGON: &[&[&[f64]]] = &[&[
        &[0.0, 0.0],
        &[1.0, 0.0],
        &[1.0, 1.0],
        &[0.0, 1.0],
        &[0.0, 0.0],
    ]];

    #[test]
    fn test_big_endian() {
        let point_wkb = Wkb::new(BE_POINT_WKB.to_vec());
        let geom: Result<Geometry<f64>, _> = point_wkb.clone().try_into();
        assert!(geom.is_err());
    }

    #[test]
    fn test_point_srid() {
        let geom_from_slice: Geometry<f64> = Wkb::from_slice(POINT_WKB_SRID).unwrap();

        let point_wkb = Wkb::new(POINT_WKB.to_vec());
        let geom: Geometry<f64> = point_wkb.clone().try_into().unwrap();
        assert_eq!(geom, geom_from_slice);
    }

    #[test]
    fn test_point() {
        let geom_from_slice: Geometry<f64> = Wkb::from_slice(POINT_WKB).unwrap();

        let point_wkb = Wkb::new(POINT_WKB.to_vec());
        let geom: Geometry<f64> = point_wkb.clone().try_into().unwrap();
        assert_eq!(geom, geom_from_slice);

        match geom {
            Geometry::Point(p) => {
                assert_eq!(p.x(), POINT[0]);
                assert_eq!(p.y(), POINT[1]);
            }
            _ => panic!("wrong type"),
        }

        let round_trip: Wkb = geom.try_into().unwrap();
        assert_eq!(round_trip.take(), point_wkb.take());
    }

    #[test]
    fn test_line_string() {
        let linestring_wkb = Wkb::new(LINESTRING_WKB.to_vec());
        let geom: Geometry<f64> = linestring_wkb.clone().try_into().unwrap();
        match geom {
            Geometry::LineString(ref l) => {
                assert_eq!(l[0].x, LINESTRING[0][0]);
                assert_eq!(l[0].y, LINESTRING[0][1]);
                assert_eq!(l[1].x, LINESTRING[1][0]);
                assert_eq!(l[1].y, LINESTRING[1][1]);
            }
            _ => panic!("wrong type"),
        }

        let round_trip: Wkb = geom.try_into().unwrap();
        assert_eq!(round_trip.take(), linestring_wkb.take());
    }

    #[test]
    fn test_polygon() {
        let polygon_wkb = Wkb::new(POLYGON_WKB.to_vec());
        let geom: Geometry<f64> = polygon_wkb.clone().try_into().unwrap();
        match geom {
            Geometry::Polygon(ref p) => {
                let ring = p.exterior();
                assert_eq!(ring[0].x, POLYGON[0][0][0]);
                assert_eq!(ring[0].y, POLYGON[0][0][1]);
                assert_eq!(ring[1].x, POLYGON[0][1][0]);
                assert_eq!(ring[1].y, POLYGON[0][1][1]);
                assert_eq!(ring[2].x, POLYGON[0][2][0]);
                assert_eq!(ring[2].y, POLYGON[0][2][1]);
                assert_eq!(ring[3].x, POLYGON[0][3][0]);
                assert_eq!(ring[3].y, POLYGON[0][3][1]);
                assert_eq!(ring[4].x, POLYGON[0][4][0]);
                assert_eq!(ring[4].y, POLYGON[0][4][1]);

                assert!(p.interiors().is_empty());
            }
            _ => panic!("wrong type"),
        }

        let round_trip: Wkb = geom.try_into().unwrap();
        assert_eq!(round_trip.take(), polygon_wkb.take());
    }

    #[test]
    fn test_polygon_with_hole() {
        let polygon_wkb = Wkb::new(POLYGON_WITH_HOLE_WKB.to_vec());
        let geom: Geometry<f64> = polygon_wkb.clone().try_into().unwrap();
        match geom {
            Geometry::Polygon(ref p) => {
                let ring = p.exterior();
                assert_eq!(ring[0].x, POLYGON_WITH_HOLE[0][0][0]);
                assert_eq!(ring[0].y, POLYGON_WITH_HOLE[0][0][1]);
                assert_eq!(ring[1].x, POLYGON_WITH_HOLE[0][1][0]);
                assert_eq!(ring[1].y, POLYGON_WITH_HOLE[0][1][1]);
                assert_eq!(ring[2].x, POLYGON_WITH_HOLE[0][2][0]);
                assert_eq!(ring[2].y, POLYGON_WITH_HOLE[0][2][1]);
                assert_eq!(ring[3].x, POLYGON_WITH_HOLE[0][3][0]);
                assert_eq!(ring[3].y, POLYGON_WITH_HOLE[0][3][1]);
                assert_eq!(ring[4].x, POLYGON_WITH_HOLE[0][4][0]);
                assert_eq!(ring[4].y, POLYGON_WITH_HOLE[0][4][1]);

                let interiors = p.interiors();
                assert!(!interiors.is_empty());
                let i0 = &interiors[0];
                assert_eq!(i0[0].x, POLYGON_WITH_HOLE[1][0][0]);
                assert_eq!(i0[0].y, POLYGON_WITH_HOLE[1][0][1]);
                assert_eq!(i0[1].x, POLYGON_WITH_HOLE[1][1][0]);
                assert_eq!(i0[1].y, POLYGON_WITH_HOLE[1][1][1]);
                assert_eq!(i0[2].x, POLYGON_WITH_HOLE[1][2][0]);
                assert_eq!(i0[2].y, POLYGON_WITH_HOLE[1][2][1]);
                assert_eq!(i0[3].x, POLYGON_WITH_HOLE[1][3][0]);
                assert_eq!(i0[3].y, POLYGON_WITH_HOLE[1][3][1]);
                assert_eq!(i0[4].x, POLYGON_WITH_HOLE[1][4][0]);
                assert_eq!(i0[4].y, POLYGON_WITH_HOLE[1][4][1]);
            }
            _ => panic!("wrong type"),
        }

        let round_trip: Wkb = geom.try_into().unwrap();
        assert_eq!(round_trip.take(), polygon_wkb.take());
    }

    #[test]
    fn test_multipoint() {
        let multipoint_wkb = Wkb::new(MULTIPOINT_WKB.to_vec());
        let geom: Geometry<f64> = multipoint_wkb.clone().try_into().unwrap();
        match geom {
            Geometry::MultiPoint(ref p) => {
                assert_eq!(p.0[0].x(), MULTIPOINT[0][0]);
                assert_eq!(p.0[0].y(), MULTIPOINT[0][1]);
                assert_eq!(p.0[1].x(), MULTIPOINT[1][0]);
                assert_eq!(p.0[1].y(), MULTIPOINT[1][1]);
            }
            _ => panic!("wrong type"),
        }

        let round_trip: Wkb = geom.try_into().unwrap();
        assert_eq!(round_trip.take(), multipoint_wkb.take());
    }

    #[test]
    fn test_multilinestring() {
        let multilinestring_wkb = Wkb::new(MULTILINESTRING_WKB.to_vec());
        let geom: Geometry<f64> = multilinestring_wkb.clone().try_into().unwrap();
        match geom {
            Geometry::MultiLineString(ref l) => {
                assert_eq!(l.0[0][0].x, MULTILINESTRING[0][0][0]);
                assert_eq!(l.0[0][0].y, MULTILINESTRING[0][0][1]);
                assert_eq!(l.0[0][1].x, MULTILINESTRING[0][1][0]);
                assert_eq!(l.0[0][1].y, MULTILINESTRING[0][1][1]);
            }
            _ => panic!("wrong type"),
        }

        let round_trip: Wkb = geom.try_into().unwrap();
        assert_eq!(round_trip.take(), multilinestring_wkb.take());
    }

    #[test]
    fn test_multipolygon() {
        let multipolygon_wkb = Wkb::new(MULTIPOLYGON_WKB.to_vec());
        let geom: Geometry<f64> = multipolygon_wkb.clone().try_into().unwrap();
        match geom {
            Geometry::MultiPolygon(ref p) => {
                let p0 = &p.0[0];
                let r0 = p0.exterior();
                assert!(p0.interiors().is_empty());
                assert_eq!(r0[0].x, MULTIPOLYGON[0][0][0][0]);
                assert_eq!(r0[0].y, MULTIPOLYGON[0][0][0][1]);
                assert_eq!(r0[1].x, MULTIPOLYGON[0][0][1][0]);
                assert_eq!(r0[1].y, MULTIPOLYGON[0][0][1][1]);
                assert_eq!(r0[2].x, MULTIPOLYGON[0][0][2][0]);
                assert_eq!(r0[2].y, MULTIPOLYGON[0][0][2][1]);
                assert_eq!(r0[3].x, MULTIPOLYGON[0][0][3][0]);
                assert_eq!(r0[3].y, MULTIPOLYGON[0][0][3][1]);
                assert_eq!(r0[4].x, MULTIPOLYGON[0][0][4][0]);
                assert_eq!(r0[4].y, MULTIPOLYGON[0][0][4][1]);

                let p1 = &p.0[1];
                let r1 = p1.exterior();
                assert!(p1.interiors().is_empty());
                assert_eq!(r1[0].x, MULTIPOLYGON[1][0][0][0]);
                assert_eq!(r1[0].y, MULTIPOLYGON[1][0][0][1]);
                assert_eq!(r1[1].x, MULTIPOLYGON[1][0][1][0]);
                assert_eq!(r1[1].y, MULTIPOLYGON[1][0][1][1]);
                assert_eq!(r1[2].x, MULTIPOLYGON[1][0][2][0]);
                assert_eq!(r1[2].y, MULTIPOLYGON[1][0][2][1]);
                assert_eq!(r1[3].x, MULTIPOLYGON[1][0][3][0]);
                assert_eq!(r1[3].y, MULTIPOLYGON[1][0][3][1]);
                assert_eq!(r1[4].x, MULTIPOLYGON[1][0][4][0]);
                assert_eq!(r1[4].y, MULTIPOLYGON[1][0][4][1]);
            }
            _ => panic!("wrong type"),
        }

        let round_trip: Wkb = geom.try_into().unwrap();
        assert_eq!(round_trip.take(), multipolygon_wkb.take());
    }

    #[test]
    fn test_geometrycollection() {
        let geometrycollection_wkb = Wkb::new(GEOMETRYCOLLECTION_WKB.to_vec());
        let geom: Geometry<f64> = geometrycollection_wkb.clone().try_into().unwrap();

        match geom {
            Geometry::GeometryCollection(ref g) => {
                let g0 = &g.0[0];
                match g0 {
                    Geometry::Point(p) => {
                        assert_eq!(p.x(), GEOMETRYCOLLECTION_POINT[0]);
                        assert_eq!(p.y(), GEOMETRYCOLLECTION_POINT[1]);
                    }
                    _ => panic!("wrong type"),
                }

                let g1 = &g.0[1];
                match g1 {
                    Geometry::Polygon(p) => {
                        let ring = p.exterior();
                        assert_eq!(ring[0].x, GEOMETRYCOLLECTION_POLYGON[0][0][0]);
                        assert_eq!(ring[0].y, GEOMETRYCOLLECTION_POLYGON[0][0][1]);
                        assert_eq!(ring[1].x, GEOMETRYCOLLECTION_POLYGON[0][1][0]);
                        assert_eq!(ring[1].y, GEOMETRYCOLLECTION_POLYGON[0][1][1]);
                        assert_eq!(ring[2].x, GEOMETRYCOLLECTION_POLYGON[0][2][0]);
                        assert_eq!(ring[2].y, GEOMETRYCOLLECTION_POLYGON[0][2][1]);
                        assert_eq!(ring[3].x, GEOMETRYCOLLECTION_POLYGON[0][3][0]);
                        assert_eq!(ring[3].y, GEOMETRYCOLLECTION_POLYGON[0][3][1]);
                        assert_eq!(ring[4].x, GEOMETRYCOLLECTION_POLYGON[0][4][0]);
                        assert_eq!(ring[4].y, GEOMETRYCOLLECTION_POLYGON[0][4][1]);

                        assert!(p.interiors().is_empty());
                    }
                    _ => panic!("wrong type"),
                }
            }
            _ => panic!("wrong type"),
        }

        let round_trip: Wkb = geom.try_into().unwrap();
        assert_eq!(round_trip.take(), geometrycollection_wkb.take());
    }
}
