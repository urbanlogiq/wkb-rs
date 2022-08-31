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
#![feature(cursor_remaining)]

use geo_types::geometry::{
    Coordinate, GeometryCollection, LineString, MultiLineString, MultiPoint, MultiPolygon, Point,
    Polygon,
};
use geo_types::Geometry;
use num_derive::FromPrimitive;
use num_traits::Float;
use num_traits::FromPrimitive;
use std::convert::TryFrom;
use std::fmt::Debug;
use std::io::Cursor;
use std::io::Read;

pub trait NumTy = Float + Debug + Into<f64> + From<f64>;

#[derive(Debug)]
pub enum WkbError {
    IoError(std::io::Error),
    Unsupported(&'static str),
    UnsupportedEndianess,
    InconsistentType,
}

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

impl<'a, 'b> LittleEndianReader<'a, 'b> {
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

pub struct Wkb(Vec<u8>);

impl Wkb {
    pub fn new(wkb: Vec<u8>) -> Self {
        Self(wkb)
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
    buffer.push(1);
    write_u32(buffer, GeomTy::Point as u32);
    write_value(buffer, p.x());
    write_value(buffer, p.y());
}

#[inline]
fn write_line_string<T: NumTy>(buffer: &mut Vec<u8>, l: &LineString<T>) {
    buffer.push(1);

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
    buffer.push(1);

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
            buffer.push(1);

            write_u32(buffer, GeomTy::MultiPoint as u32);
            write_u32(buffer, mp.0.len() as u32);

            mp.iter().for_each(|p| write_point(buffer, p));
        }
        Geometry::MultiLineString(ml) => {
            buffer.push(1);

            write_u32(buffer, GeomTy::MultiLineString as u32);
            write_u32(buffer, ml.0.len() as u32);

            ml.iter().for_each(|l| write_line_string(buffer, l));
        }
        Geometry::MultiPolygon(mp) => {
            buffer.push(1);

            write_u32(buffer, GeomTy::MultiPolygon as u32);
            write_u32(buffer, mp.0.len() as u32);

            mp.iter().for_each(|p| write_polygon(buffer, p));
        }
        Geometry::GeometryCollection(gc) => {
            buffer.push(1);

            write_u32(buffer, GeomTy::GeometryCollection as u32);
            write_u32(buffer, gc.len() as u32);

            gc.iter().for_each(|g| write_geometry(buffer, g));
        }
        _ => unimplemented!(),
    }
}

impl<T: NumTy> TryFrom<Geometry<T>> for Wkb {
    type Error = WkbError;

    fn try_from(geom: Geometry<T>) -> Result<Wkb, Self::Error> {
        let mut buffer = Vec::new();

        write_geometry(&mut buffer, &geom);

        Ok(Self(buffer))
    }
}

fn read_coordinate<T: NumTy>(reader: &mut LittleEndianReader) -> Result<Coordinate<T>, WkbError> {
    let x: T = reader.read_f64()?.into();
    let y: T = reader.read_f64()?.into();

    Ok(Coordinate { x, y })
}

fn read_coordinates<T: NumTy>(
    reader: &mut LittleEndianReader,
) -> Result<Vec<Coordinate<T>>, WkbError> {
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

    let endianness = Endian::from_u8(endianness[0])
        .ok_or(WkbError::Unsupported("unsupported endianness value"))?;
    let mut reader = match endianness {
        Endian::Big => return Err(WkbError::UnsupportedEndianess),
        Endian::Little => LittleEndianReader { cursor },
    };

    let ty = GeomTy::from_u32(reader.read_u32()?)
        .ok_or(WkbError::Unsupported("unsupported type value"))?;
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

    fn try_from(geom: Wkb) -> Result<Geometry<T>, Self::Error> {
        let mut cursor = Cursor::new(geom.0.as_slice());
        read_wkb(&mut cursor)
    }
}
